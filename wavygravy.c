/*
 * wavygravy.c -- straight C translation of wavygravy.py
 *                hide a short UTF-8 message in an audio file, recoverable
 *                after telephone / HAM-radio grade degradation.
 *
 * Single file, no third-party dependencies (libc + libm + POSIX only).
 * Everything that wavygravy.py pulled from numpy / scipy / cryptography /
 * hashlib is implemented below: SHA-256, HMAC, PBKDF2, scrypt, ChaCha20,
 * Poly1305, ChaCha20-Poly1305, FFT (radix-2 + Bluestein), FIR design,
 * resample_poly, Hilbert transform, WAV I/O.
 *
 * The on-air format is unchanged (same KDF salts, key labels, FEC, framing,
 * carrier, sync words), so files made by wavygravy.py decode here and
 * vice-versa.
 *
 * Build:
 *     cc -O2 -o wavygravy wavygravy.c -lm
 *
 * Usage (same as the original):
 *     WAVYGRAVY_PASSWORD=... wavygravy encode  cover.wav "stop spying on me" out.wav
 *     WAVYGRAVY_PASSWORD=... wavygravy decode  out.wav [message.txt]
 *                         wavygravy capacity cover.wav
 *
 * If WAVYGRAVY_PASSWORD is unset the password is prompted for (not echoed).
 *
 * Differences from the Python version (unavoidable without libraries):
 *   - WAV: reads PCM 8/16/24/32-bit and float32/64 itself; anything else is
 *     handed to an external `ffmpeg` if one is installed (as the original did).
 *   - --simulate uses its own Gaussian noise generator, not numpy's PCG64.
 *   - POSIX only (/dev/urandom, termios, fork/exec for ffmpeg).
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE  /* glibc: keep BSD/POSIX extras visible */
#define _DARWIN_C_SOURCE /* macOS: _POSIX_C_SOURCE alone hides mkdtemp & co. */
#include <complex.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#define PI 3.14159265358979323846

/* POSIX 2008; declared here too so strict -std=c99 builds on any libc still see it */
char *mkdtemp(char *template_);

/* ── constants ────────────────────────────────────────────────────────────── */

#define WORK_FS 8000 /* internal rate (telephone rate) */
#define SYNC_BITS 8  /* keyed sync symbols per block */
#define DATA_BITS 56 /* coded data symbols per block */
#define BLOCK (SYNC_BITS + DATA_BITS)
#define LEN_BYTES 8
#define SALT_BYTES 8
#define HDR_BYTES (LEN_BYTES + SALT_BYTES)
#define TAG_BYTES 16
#define ENV_WIN 400 /* 50 ms @ 8 kHz */
#define ENV_FLOOR 0.1
#define SYNC_Z 7.0
#define FREQ_STEP 2.5
#define KDF_N 16384u
#define DEFAULT_STRENGTH_DB 18.0

static const char *RED_NAME[3] = {"low", "medium", "high"};
static const int RED_L[3] = {256, 512, 1024};

typedef double complex cx;

/* ── error handling / small utils ─────────────────────────────────────────── */

static char g_err[512];

/* record error message, return -1 */
static int fail(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof g_err, fmt, ap);
    va_end(ap);
    return -1;
}

static void *xmalloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p)
    {
        fprintf(stderr, "wavygravy: out of memory\n");
        exit(2);
    }
    return p;
}
static void *xcalloc(size_t n, size_t s)
{
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p)
    {
        fprintf(stderr, "wavygravy: out of memory\n");
        exit(2);
    }
    return p;
}
static void *xrealloc(void *p, size_t n)
{
    p = realloc(p, n ? n : 1);
    if (!p)
    {
        fprintf(stderr, "wavygravy: out of memory\n");
        exit(2);
    }
    return p;
}

static inline uint32_t le32(const uint8_t *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static inline void put_le32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}
static inline void put_le64(uint8_t *p, uint64_t v)
{
    put_le32(p, (uint32_t)v);
    put_le32(p + 4, (uint32_t)(v >> 32));
}
static inline uint32_t be32(const uint8_t *p)
{
    return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | (uint32_t)p[3];
}
static inline void put_be32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)v;
}
static inline uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }

/* ═══════════════════════════════════════════════════════════════════════════
 *  crypto primitives
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── SHA-256 ── */

typedef struct
{
    uint32_t h[8];
    uint64_t total;
    uint8_t buf[64];
    size_t blen;
} sha256_ctx;

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define ROL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

static void sha256_compress(uint32_t h[8], const uint8_t *p)
{
    uint32_t w[64], a, b, c, d, e, f, g, hh, t1, t2;
    for (int i = 0; i < 16; i++)
        w[i] = be32(p + 4 * i);
    for (int i = 16; i < 64; i++)
    {
        uint32_t s0 = ROR32(w[i - 15], 7) ^ ROR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROR32(w[i - 2], 17) ^ ROR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    a = h[0];
    b = h[1];
    c = h[2];
    d = h[3];
    e = h[4];
    f = h[5];
    g = h[6];
    hh = h[7];
    for (int i = 0; i < 64; i++)
    {
        uint32_t S1 = ROR32(e, 6) ^ ROR32(e, 11) ^ ROR32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        t1 = hh + S1 + ch + K256[i] + w[i];
        uint32_t S0 = ROR32(a, 2) ^ ROR32(a, 13) ^ ROR32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        t2 = S0 + mj;
        hh = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
}

static void sha256_init(sha256_ctx *c)
{
    static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    memcpy(c->h, iv, sizeof iv);
    c->total = 0;
    c->blen = 0;
}
static void sha256_update(sha256_ctx *c, const uint8_t *p, size_t n)
{
    c->total += n;
    while (n)
    {
        size_t k = 64 - c->blen;
        if (k > n)
            k = n;
        memcpy(c->buf + c->blen, p, k);
        c->blen += k;
        p += k;
        n -= k;
        if (c->blen == 64)
        {
            sha256_compress(c->h, c->buf);
            c->blen = 0;
        }
    }
}
static void sha256_final(sha256_ctx *c, uint8_t out[32])
{
    uint64_t bits = c->total * 8;
    uint8_t pad[72] = {0x80};
    size_t padlen = (c->blen < 56) ? 56 - c->blen : 120 - c->blen;
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++)
        lenb[i] = (uint8_t)(bits >> (56 - 8 * i));
    sha256_update(c, pad, padlen);
    sha256_update(c, lenb, 8);
    for (int i = 0; i < 8; i++)
        put_be32(out + 4 * i, c->h[i]);
}

static void hmac_sha256(const uint8_t *key, size_t klen, const uint8_t *msg, size_t mlen, uint8_t out[32])
{
    uint8_t k[64] = {0}, ipad[64], opad[64], inner[32];
    sha256_ctx c;
    if (klen > 64)
    {
        sha256_init(&c);
        sha256_update(&c, key, klen);
        sha256_final(&c, k);
    }
    else
        memcpy(k, key, klen);
    for (int i = 0; i < 64; i++)
    {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }
    sha256_init(&c);
    sha256_update(&c, ipad, 64);
    sha256_update(&c, msg, mlen);
    sha256_final(&c, inner);
    sha256_init(&c);
    sha256_update(&c, opad, 64);
    sha256_update(&c, inner, 32);
    sha256_final(&c, out);
}

static void pbkdf2_sha256(const uint8_t *pw, size_t pwlen, const uint8_t *salt, size_t saltlen,
                          uint32_t iters, uint8_t *dk, size_t dklen)
{
    uint8_t *sb = xmalloc(saltlen + 4);
    memcpy(sb, salt, saltlen);
    for (uint32_t blk = 1; dklen > 0; blk++)
    {
        uint8_t u[32], t[32];
        put_be32(sb + saltlen, blk);
        hmac_sha256(pw, pwlen, sb, saltlen + 4, u);
        memcpy(t, u, 32);
        for (uint32_t i = 1; i < iters; i++)
        {
            hmac_sha256(pw, pwlen, u, 32, u);
            for (int j = 0; j < 32; j++)
                t[j] ^= u[j];
        }
        size_t k = dklen < 32 ? dklen : 32;
        memcpy(dk, t, k);
        dk += k;
        dklen -= k;
    }
    free(sb);
}

/* ── scrypt (RFC 7914) ── */

static void salsa20_8(uint32_t B[16])
{
    uint32_t x[16];
    memcpy(x, B, 64);
    for (int i = 0; i < 8; i += 2)
    {
        x[4] ^= ROL32(x[0] + x[12], 7);
        x[8] ^= ROL32(x[4] + x[0], 9);
        x[12] ^= ROL32(x[8] + x[4], 13);
        x[0] ^= ROL32(x[12] + x[8], 18);
        x[9] ^= ROL32(x[5] + x[1], 7);
        x[13] ^= ROL32(x[9] + x[5], 9);
        x[1] ^= ROL32(x[13] + x[9], 13);
        x[5] ^= ROL32(x[1] + x[13], 18);
        x[14] ^= ROL32(x[10] + x[6], 7);
        x[2] ^= ROL32(x[14] + x[10], 9);
        x[6] ^= ROL32(x[2] + x[14], 13);
        x[10] ^= ROL32(x[6] + x[2], 18);
        x[3] ^= ROL32(x[15] + x[11], 7);
        x[7] ^= ROL32(x[3] + x[15], 9);
        x[11] ^= ROL32(x[7] + x[3], 13);
        x[15] ^= ROL32(x[11] + x[7], 18);
        x[1] ^= ROL32(x[0] + x[3], 7);
        x[2] ^= ROL32(x[1] + x[0], 9);
        x[3] ^= ROL32(x[2] + x[1], 13);
        x[0] ^= ROL32(x[3] + x[2], 18);
        x[6] ^= ROL32(x[5] + x[4], 7);
        x[7] ^= ROL32(x[6] + x[5], 9);
        x[4] ^= ROL32(x[7] + x[6], 13);
        x[5] ^= ROL32(x[4] + x[7], 18);
        x[11] ^= ROL32(x[10] + x[9], 7);
        x[8] ^= ROL32(x[11] + x[10], 9);
        x[9] ^= ROL32(x[8] + x[11], 13);
        x[10] ^= ROL32(x[9] + x[8], 18);
        x[12] ^= ROL32(x[15] + x[14], 7);
        x[13] ^= ROL32(x[12] + x[15], 9);
        x[14] ^= ROL32(x[13] + x[12], 13);
        x[15] ^= ROL32(x[14] + x[13], 18);
    }
    for (int i = 0; i < 16; i++)
        B[i] += x[i];
}

static void scrypt_blockmix(uint32_t *B, uint32_t *Y, uint32_t r)
{
    uint32_t X[16];
    memcpy(X, &B[(2 * r - 1) * 16], 64);
    for (uint32_t i = 0; i < 2 * r; i++)
    {
        for (int j = 0; j < 16; j++)
            X[j] ^= B[i * 16 + j];
        salsa20_8(X);
        memcpy(&Y[i * 16], X, 64);
    }
    for (uint32_t i = 0; i < r; i++)
        memcpy(&B[i * 16], &Y[(2 * i) * 16], 64);
    for (uint32_t i = 0; i < r; i++)
        memcpy(&B[(i + r) * 16], &Y[(2 * i + 1) * 16], 64);
}

static void scrypt_kdf(const uint8_t *pw, size_t pwlen, const uint8_t *salt, size_t saltlen,
                       uint32_t N, uint32_t r, uint32_t p, uint8_t *out, size_t outlen)
{
    size_t blen = 128 * (size_t)r, w = 32 * (size_t)r;
    uint8_t *B = xmalloc(p * blen);
    uint32_t *X = xmalloc(blen), *Y = xmalloc(blen), *V = xmalloc(blen * N);
    pbkdf2_sha256(pw, pwlen, salt, saltlen, 1, B, p * blen);
    for (uint32_t pi = 0; pi < p; pi++)
    {
        uint8_t *Bp = B + pi * blen;
        for (size_t k = 0; k < w; k++)
            X[k] = le32(Bp + 4 * k);
        for (uint32_t i = 0; i < N; i++)
        {
            memcpy(V + (size_t)i * w, X, blen);
            scrypt_blockmix(X, Y, r);
        }
        for (uint32_t i = 0; i < N; i++)
        {
            uint32_t j = X[(2 * r - 1) * 16] & (N - 1);
            for (size_t k = 0; k < w; k++)
                X[k] ^= V[(size_t)j * w + k];
            scrypt_blockmix(X, Y, r);
        }
        for (size_t k = 0; k < w; k++)
            put_le32(Bp + 4 * k, X[k]);
    }
    pbkdf2_sha256(pw, pwlen, B, p * blen, 1, out, outlen);
    free(B);
    free(X);
    free(Y);
    free(V);
}

/* ── ChaCha20 / Poly1305 / ChaCha20-Poly1305 (RFC 8439) ── */

#define QR(a, b, c, d) \
    a += b;            \
    d ^= a;            \
    d = ROL32(d, 16);  \
    c += d;            \
    b ^= c;            \
    b = ROL32(b, 12);  \
    a += b;            \
    d ^= a;            \
    d = ROL32(d, 8);   \
    c += d;            \
    b ^= c;            \
    b = ROL32(b, 7);

static void chacha20_block(const uint8_t key[32], uint32_t ctr, const uint8_t nonce[12], uint8_t out[64])
{
    uint32_t s[16], x[16];
    s[0] = 0x61707865;
    s[1] = 0x3320646e;
    s[2] = 0x79622d32;
    s[3] = 0x6b206574;
    for (int i = 0; i < 8; i++)
        s[4 + i] = le32(key + 4 * i);
    s[12] = ctr;
    for (int i = 0; i < 3; i++)
        s[13 + i] = le32(nonce + 4 * i);
    memcpy(x, s, 64);
    for (int i = 0; i < 10; i++)
    {
        QR(x[0], x[4], x[8], x[12])
        QR(x[1], x[5], x[9], x[13])
            QR(x[2], x[6], x[10], x[14]) QR(x[3], x[7], x[11], x[15])
                QR(x[0], x[5], x[10], x[15]) QR(x[1], x[6], x[11], x[12])
                    QR(x[2], x[7], x[8], x[13]) QR(x[3], x[4], x[9], x[14])
    }
    for (int i = 0; i < 16; i++)
        put_le32(out + 4 * i, x[i] + s[i]);
}

/* XOR buf with the ChaCha20 stream starting at block counter `ctr` */
static void chacha20_xor(const uint8_t key[32], uint32_t ctr, const uint8_t nonce[12], uint8_t *buf, size_t n)
{
    uint8_t ks[64];
    for (size_t off = 0; off < n; off += 64, ctr++)
    {
        chacha20_block(key, ctr, nonce, ks);
        size_t k = n - off < 64 ? n - off : 64;
        for (size_t i = 0; i < k; i++)
            buf[off + i] ^= ks[i];
    }
}

/* Poly1305 over a message whose length is a multiple of 16 (5x26-bit limbs) */
static void poly1305_mac(const uint8_t key[32], const uint8_t *m, size_t len, uint8_t tag[16])
{
    uint32_t r0 = le32(key) & 0x3ffffff;
    uint32_t r1 = (le32(key + 3) >> 2) & 0x3ffff03;
    uint32_t r2 = (le32(key + 6) >> 4) & 0x3ffc0ff;
    uint32_t r3 = (le32(key + 9) >> 6) & 0x3f03fff;
    uint32_t r4 = (le32(key + 12) >> 8) & 0x00fffff;
    uint32_t s1 = r1 * 5, s2 = r2 * 5, s3 = r3 * 5, s4 = r4 * 5;
    uint32_t h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0;
    for (; len >= 16; len -= 16, m += 16)
    {
        h0 += le32(m) & 0x3ffffff;
        h1 += (le32(m + 3) >> 2) & 0x3ffffff;
        h2 += (le32(m + 6) >> 4) & 0x3ffffff;
        h3 += (le32(m + 9) >> 6) & 0x3ffffff;
        h4 += (le32(m + 12) >> 8) | (1u << 24);
        uint64_t d0 = (uint64_t)h0 * r0 + (uint64_t)h1 * s4 + (uint64_t)h2 * s3 + (uint64_t)h3 * s2 + (uint64_t)h4 * s1;
        uint64_t d1 = (uint64_t)h0 * r1 + (uint64_t)h1 * r0 + (uint64_t)h2 * s4 + (uint64_t)h3 * s3 + (uint64_t)h4 * s2;
        uint64_t d2 = (uint64_t)h0 * r2 + (uint64_t)h1 * r1 + (uint64_t)h2 * r0 + (uint64_t)h3 * s4 + (uint64_t)h4 * s3;
        uint64_t d3 = (uint64_t)h0 * r3 + (uint64_t)h1 * r2 + (uint64_t)h2 * r1 + (uint64_t)h3 * r0 + (uint64_t)h4 * s4;
        uint64_t d4 = (uint64_t)h0 * r4 + (uint64_t)h1 * r3 + (uint64_t)h2 * r2 + (uint64_t)h3 * r1 + (uint64_t)h4 * r0;
        uint32_t c;
        c = (uint32_t)(d0 >> 26);
        h0 = (uint32_t)d0 & 0x3ffffff;
        d1 += c;
        c = (uint32_t)(d1 >> 26);
        h1 = (uint32_t)d1 & 0x3ffffff;
        d2 += c;
        c = (uint32_t)(d2 >> 26);
        h2 = (uint32_t)d2 & 0x3ffffff;
        d3 += c;
        c = (uint32_t)(d3 >> 26);
        h3 = (uint32_t)d3 & 0x3ffffff;
        d4 += c;
        c = (uint32_t)(d4 >> 26);
        h4 = (uint32_t)d4 & 0x3ffffff;
        h0 += c * 5;
        c = h0 >> 26;
        h0 &= 0x3ffffff;
        h1 += c;
    }
    uint32_t c;
    c = h1 >> 26;
    h1 &= 0x3ffffff;
    h2 += c;
    c = h2 >> 26;
    h2 &= 0x3ffffff;
    h3 += c;
    c = h3 >> 26;
    h3 &= 0x3ffffff;
    h4 += c;
    c = h4 >> 26;
    h4 &= 0x3ffffff;
    h0 += c * 5;
    c = h0 >> 26;
    h0 &= 0x3ffffff;
    h1 += c;
    uint32_t g0 = h0 + 5;
    c = g0 >> 26;
    g0 &= 0x3ffffff;
    uint32_t g1 = h1 + c;
    c = g1 >> 26;
    g1 &= 0x3ffffff;
    uint32_t g2 = h2 + c;
    c = g2 >> 26;
    g2 &= 0x3ffffff;
    uint32_t g3 = h3 + c;
    c = g3 >> 26;
    g3 &= 0x3ffffff;
    uint32_t g4 = h4 + c - (1u << 26);
    uint32_t mask = (g4 >> 31) - 1;
    g0 &= mask;
    g1 &= mask;
    g2 &= mask;
    g3 &= mask;
    g4 &= mask;
    mask = ~mask;
    h0 = (h0 & mask) | g0;
    h1 = (h1 & mask) | g1;
    h2 = (h2 & mask) | g2;
    h3 = (h3 & mask) | g3;
    h4 = (h4 & mask) | g4;
    uint32_t t0 = (h0) | (h1 << 26);
    uint32_t t1 = (h1 >> 6) | (h2 << 20);
    uint32_t t2 = (h2 >> 12) | (h3 << 14);
    uint32_t t3 = (h3 >> 18) | (h4 << 8);
    uint64_t f;
    f = (uint64_t)t0 + le32(key + 16);
    put_le32(tag, (uint32_t)f);
    f = (uint64_t)t1 + le32(key + 20) + (f >> 32);
    put_le32(tag + 4, (uint32_t)f);
    f = (uint64_t)t2 + le32(key + 24) + (f >> 32);
    put_le32(tag + 8, (uint32_t)f);
    f = (uint64_t)t3 + le32(key + 28) + (f >> 32);
    put_le32(tag + 12, (uint32_t)f);
}

static void aead_tag(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *aad, size_t aadlen,
                     const uint8_t *ct, size_t ctlen, uint8_t tag[16])
{
    uint8_t otk[64];
    chacha20_block(key, 0, nonce, otk);
    size_t ap = (16 - aadlen % 16) % 16, cp = (16 - ctlen % 16) % 16;
    size_t mlen = aadlen + ap + ctlen + cp + 16;
    uint8_t *m = xcalloc(mlen, 1);
    memcpy(m, aad, aadlen);
    memcpy(m + aadlen + ap, ct, ctlen);
    put_le64(m + aadlen + ap + ctlen + cp, aadlen);
    put_le64(m + aadlen + ap + ctlen + cp + 8, ctlen);
    poly1305_mac(otk, m, mlen, tag);
    free(m);
}

/* out must hold n + 16 bytes */
static void aead_encrypt(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *aad, size_t aadlen,
                         const uint8_t *pt, size_t n, uint8_t *out)
{
    memcpy(out, pt, n);
    chacha20_xor(key, 1, nonce, out, n);
    aead_tag(key, nonce, aad, aadlen, out, n, out + n);
}

/* ct holds n + 16 bytes (cipher + tag); returns 0 and fills pt (n bytes) on success, -1 on bad tag */
static int aead_decrypt(const uint8_t key[32], const uint8_t nonce[12], const uint8_t *aad, size_t aadlen,
                        const uint8_t *ct, size_t n, uint8_t *pt)
{
    uint8_t tag[16], diff = 0;
    aead_tag(key, nonce, aad, aadlen, ct, n, tag);
    for (int i = 0; i < 16; i++)
        diff |= tag[i] ^ ct[n + i];
    if (diff)
        return -1;
    memcpy(pt, ct, n);
    chacha20_xor(key, 1, nonce, pt, n);
    return 0;
}

/* ── keys & keyed pseudo-randomness ───────────────────────────────────────── */

static const uint8_t ZERO_NONCE[12] = {0};

/* Key for the carrier / sync / interleaver. Depends on the password only. */
static void carrier_key(const char *pw, uint8_t out[32])
{
    static const char salt[] = "wavygravy/carrier/v1";
    scrypt_kdf((const uint8_t *)pw, strlen(pw), (const uint8_t *)salt, sizeof salt - 1, KDF_N, 8, 1, out, 32);
}
static void msg_key(const char *pw, const uint8_t salt[SALT_BYTES], uint8_t out[32])
{
    scrypt_kdf((const uint8_t *)pw, strlen(pw), salt, SALT_BYTES, KDF_N, 8, 1, out, 32);
}
static void subkey(const uint8_t ck[32], const char *label, uint8_t out[32])
{
    hmac_sha256(ck, 32, (const uint8_t *)label, strlen(label), out);
}
static uint8_t *keystream(const uint8_t key32[32], size_t nbytes)
{
    uint8_t *b = xcalloc(nbytes, 1);
    chacha20_xor(key32, 0, ZERO_NONCE, b, nbytes);
    return b;
}
/* n bits, one per byte (0/1), MSB-first like np.unpackbits */
static uint8_t *keyed_bits(const uint8_t key32[32], size_t n)
{
    uint8_t *ks = keystream(key32, (n + 7) / 8), *bits = xmalloc(n);
    for (size_t i = 0; i < n; i++)
        bits[i] = (ks[i >> 3] >> (7 - (i & 7))) & 1;
    free(ks);
    return bits;
}

typedef struct
{
    uint32_t k, i;
} kv_t;
static int kv_cmp(const void *a, const void *b)
{
    const kv_t *x = a, *y = b;
    if (x->k != y->k)
        return x->k < y->k ? -1 : 1;
    return x->i < y->i ? -1 : (x->i > y->i);
}
/* Deterministic permutation: stable argsort of keystream words. */
static uint32_t *keyed_perm(const uint8_t ck[32], const char *label, size_t n)
{
    uint8_t sk[32];
    subkey(ck, label, sk);
    uint8_t *ks = keystream(sk, 4 * n);
    kv_t *a = xmalloc(n * sizeof *a);
    uint32_t *perm = xmalloc(n * sizeof *perm);
    for (size_t i = 0; i < n; i++)
    {
        a[i].k = le32(ks + 4 * i);
        a[i].i = (uint32_t)i;
    }
    qsort(a, n, sizeof *a, kv_cmp);
    for (size_t i = 0; i < n; i++)
        perm[i] = a[i].i;
    free(a);
    free(ks);
    return perm;
}
static void sync_word(const uint8_t ck[32], double sw[SYNC_BITS])
{
    uint8_t sk[32];
    subkey(ck, "sync", sk);
    uint8_t *b = keyed_bits(sk, SYNC_BITS);
    for (int i = 0; i < SYNC_BITS; i++)
        sw[i] = 1.0 - 2.0 * b[i];
    free(b);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  FFT (radix-2, plus Bluestein for arbitrary lengths)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Twiddle tables for every power-of-two size up to g_tw_n: the table for size m
 * holds exp(-2 pi i j / m), j < m/2, starting at complex index m/2 - 1. */
static double *g_tw = NULL;
static size_t g_tw_n = 0;

static const double *twiddles(size_t n)
{
    if (n > g_tw_n)
    {
        free(g_tw);
        g_tw = xmalloc(n * 2 * sizeof(double) + 32);
        double *top = g_tw + 2 * (n / 2 - 1);
        for (size_t j = 0; j < n / 2; j++)
        {
            double ang = 2.0 * PI * (double)j / (double)n;
            top[2 * j] = cos(ang);
            top[2 * j + 1] = -sin(ang);
        }
        for (size_t m = n / 2; m >= 2; m >>= 1)
        {
            double *t = g_tw + 2 * (m / 2 - 1), *p = g_tw + 2 * (m - 1);
            for (size_t j = 0; j < m / 2; j++)
            {
                t[2 * j] = p[4 * j];
                t[2 * j + 1] = p[4 * j + 1];
            }
        }
        g_tw_n = n;
    }
    return g_tw;
}

static int is_pow2(size_t n) { return n && !(n & (n - 1)); }
static size_t next_pow2(size_t n)
{
    size_t m = 1;
    while (m < n)
        m <<= 1;
    return m;
}

#define FFT_BLK 4096 /* sub-transform size that stays in cache */

/* one decimation-in-frequency stage on blocks of size m over `len` elements */
static void stage_dif(double *d, size_t len, size_t m, const double *tw)
{
    size_t h = m >> 1;
    const double *t = tw + 2 * (h - 1);
    for (size_t b = 0; b < len; b += m)
    {
        double *u = d + 2 * b, *v = u + 2 * h;
        for (size_t j = 0; j < h; j++)
        {
            double ur = u[2 * j], ui = u[2 * j + 1], vr = v[2 * j], vi = v[2 * j + 1];
            double dr = ur - vr, di = ui - vi, wr = t[2 * j], wi = t[2 * j + 1];
            u[2 * j] = ur + vr;
            u[2 * j + 1] = ui + vi;
            v[2 * j] = dr * wr - di * wi;
            v[2 * j + 1] = dr * wi + di * wr;
        }
    }
}
/* one decimation-in-time stage with conjugated twiddles (inverse transform) */
static void stage_dit_inv(double *d, size_t len, size_t m, const double *tw)
{
    size_t h = m >> 1;
    const double *t = tw + 2 * (h - 1);
    for (size_t b = 0; b < len; b += m)
    {
        double *u = d + 2 * b, *v = u + 2 * h;
        for (size_t j = 0; j < h; j++)
        {
            double wr = t[2 * j], wi = t[2 * j + 1];
            double xr = v[2 * j], xi = v[2 * j + 1];
            double tr = xr * wr + xi * wi, ti = xi * wr - xr * wi;
            double ur = u[2 * j], ui = u[2 * j + 1];
            u[2 * j] = ur + tr;
            u[2 * j + 1] = ui + ti;
            v[2 * j] = ur - tr;
            v[2 * j + 1] = ui - ti;
        }
    }
}

/* forward FFT, natural order in, BIT-REVERSED order out (no permutation pass) */
static void fft_fwd_brev(cx *a, size_t n)
{
    if (n < 2)
        return;
    double *d = (double *)a;
    const double *tw = twiddles(n);
    size_t blk = n < FFT_BLK ? n : FFT_BLK;
    for (size_t m = n; m > blk; m >>= 1)
        stage_dif(d, n, m, tw);
    for (size_t s = 0; s < n; s += blk)
        for (size_t m = blk; m >= 2; m >>= 1)
            stage_dif(d + 2 * s, blk, m, tw);
}
/* inverse FFT (1/n scaled), BIT-REVERSED order in, natural order out */
static void fft_inv_from_brev(cx *a, size_t n)
{
    if (n < 2)
        return;
    double *d = (double *)a;
    const double *tw = twiddles(n);
    size_t blk = n < FFT_BLK ? n : FFT_BLK;
    for (size_t s = 0; s < n; s += blk)
        for (size_t m = 2; m <= blk; m <<= 1)
            stage_dit_inv(d + 2 * s, blk, m, tw);
    for (size_t m = 2 * blk; m <= n; m <<= 1)
        stage_dit_inv(d, n, m, tw);
    double sc = 1.0 / (double)n;
    for (size_t i = 0; i < 2 * n; i++)
        d[i] *= sc;
}
static void bit_reverse_permute(cx *a, size_t n)
{
    for (size_t i = 1, j = 0; i < n; i++)
    {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
        {
            cx t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}

/* natural-order in-place FFT; inv = 1 gives the 1/n-scaled inverse (numpy convention) */
static void fft_pow2(cx *a, size_t n, int inv)
{
    if (inv)
    {
        bit_reverse_permute(a, n);
        fft_inv_from_brev(a, n);
    }
    else
    {
        fft_fwd_brev(a, n);
        bit_reverse_permute(a, n);
    }
}

/* arbitrary length (Bluestein chirp-z) */
static void fft_any(cx *a, size_t n, int inv)
{
    if (n < 2)
        return;
    if (is_pow2(n))
    {
        fft_pow2(a, n, inv);
        return;
    }
    size_t m = next_pow2(2 * n - 1);
    cx *w = xmalloc(n * sizeof *w), *A = xcalloc(m, sizeof *A), *B = xcalloc(m, sizeof *B);
    for (size_t k = 0; k < n; k++)
    {
        uint64_t kk = ((uint64_t)k * k) % (2 * (uint64_t)n);
        double ang = -PI * (double)kk / (double)n;
        w[k] = cos(ang) + sin(ang) * I;
    }
    if (inv)
        for (size_t k = 0; k < n; k++)
            a[k] = conj(a[k]);
    for (size_t k = 0; k < n; k++)
        A[k] = a[k] * w[k];
    B[0] = conj(w[0]);
    for (size_t k = 1; k < n; k++)
        B[k] = B[m - k] = conj(w[k]);
    fft_pow2(A, m, 0);
    fft_pow2(B, m, 0);
    for (size_t k = 0; k < m; k++)
        A[k] *= B[k];
    fft_pow2(A, m, 1);
    for (size_t k = 0; k < n; k++)
        a[k] = A[k] * w[k];
    if (inv)
    {
        double s = 1.0 / (double)n;
        for (size_t k = 0; k < n; k++)
            a[k] = conj(a[k]) * s;
    }
    free(w);
    free(A);
    free(B);
}

/* scipy.signal.hilbert: analytic signal, FFT of exactly n points */
static cx *hilbert(const double *x, size_t n)
{
    cx *a = xmalloc(n * sizeof *a);
    for (size_t i = 0; i < n; i++)
        a[i] = x[i];
    fft_any(a, n, 0);
    if (n % 2 == 0)
    {
        for (size_t k = 1; k < n / 2; k++)
            a[k] *= 2.0;
        for (size_t k = n / 2 + 1; k < n; k++)
            a[k] = 0;
    }
    else
    {
        for (size_t k = 1; k < (n + 1) / 2; k++)
            a[k] *= 2.0;
        for (size_t k = (n + 1) / 2; k < n; k++)
            a[k] = 0;
    }
    fft_any(a, n, 1);
    return a;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  DSP helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static double sinc(double x) { return x == 0.0 ? 1.0 : sin(PI * x) / (PI * x); }

static double bessel_i0(double x)
{
    double sum = 1.0, term = 1.0, q = x * x / 4.0;
    for (int k = 1; k < 500; k++)
    {
        term *= q / ((double)k * k);
        sum += term;
        if (term < 1e-18 * sum)
            break;
    }
    return sum;
}

/* scipy.signal.firwin(numtaps, [left, right]) with left/right relative to
 * Nyquist, scaled to unity gain at the passband centre (DC if left == 0).
 * kaiser_beta > 0 selects a Kaiser window, otherwise Hamming. */
static double *firwin(int numtaps, double left, double right, double kaiser_beta)
{
    double *h = xmalloc(numtaps * sizeof *h);
    double alpha = 0.5 * (numtaps - 1), s = 0.0, fc;
    fc = (left == 0.0) ? 0.0 : 0.5 * (left + right);
    for (int i = 0; i < numtaps; i++)
    {
        double m = i - alpha, win;
        if (kaiser_beta > 0.0)
        {
            double r = (alpha != 0.0) ? (m / alpha) : 0.0, v = 1.0 - r * r;
            win = bessel_i0(kaiser_beta * sqrt(v > 0.0 ? v : 0.0)) / bessel_i0(kaiser_beta);
        }
        else
        {
            win = 0.54 - 0.46 * cos(2.0 * PI * i / (numtaps - 1));
        }
        h[i] = (right * sinc(right * m) - left * sinc(left * m)) * win;
        s += h[i] * cos(PI * m * fc);
    }
    for (int i = 0; i < numtaps; i++)
        h[i] /= s;
    return h;
}

/* np/scipy fftconvolve(x, h, mode="same"), done as overlap-add */
static double *conv_same(const double *x, size_t nx, const double *h, size_t nh)
{
    double *out = xcalloc(nx, sizeof *out);
    if (!nx)
        return out;
    size_t F = next_pow2(4 * nh);
    if (F < 1024)
        F = 1024;
    size_t B = F - nh + 1, start = (nh - 1) / 2;
    cx *H = xcalloc(F, sizeof *H), *buf = xmalloc(F * sizeof *buf);
    for (size_t i = 0; i < nh; i++)
        H[i] = h[i];
    fft_pow2(H, F, 0);
    for (size_t s = 0; s < nx; s += B)
    {
        size_t len = nx - s < B ? nx - s : B;
        for (size_t i = 0; i < F; i++)
            buf[i] = i < len ? x[s + i] : 0.0;
        fft_pow2(buf, F, 0);
        for (size_t i = 0; i < F; i++)
            buf[i] *= H[i];
        fft_pow2(buf, F, 1);
        for (size_t j = 0; j < len + nh - 1; j++)
        {
            size_t full = s + j;
            if (full >= start && full - start < nx)
                out[full - start] += creal(buf[j]);
        }
    }
    free(H);
    free(buf);
    return out;
}

static double *bp_fir(const double band[2], int *nh)
{
    *nh = 401;
    return firwin(401, band[0] / (WORK_FS / 2.0), band[1] / (WORK_FS / 2.0), 0.0);
}
static double *bandpass(const double *x, size_t n, const double band[2])
{
    int nh;
    double *h = bp_fir(band, &nh);
    double *y = conv_same(x, n, h, nh);
    free(h);
    return y;
}

static long gcd_l(long a, long b)
{
    while (b)
    {
        long t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/* scipy.signal.resample_poly(x, fs_out/g, fs_in/g), default kaiser(5.0) window.
 * Always returns a fresh buffer. */
static double *resample(const double *x, size_t n, long fs_in, long fs_out, size_t *nout)
{
    if (fs_in == fs_out)
    {
        double *c = xmalloc(n * sizeof *c);
        memcpy(c, x, n * sizeof *c);
        *nout = n;
        return c;
    }
    long g = gcd_l(fs_in, fs_out), up = fs_out / g, down = fs_in / g;
    long long maxr = up > down ? up : down;
    long long hl = 10 * maxr;
    int nt = (int)(2 * hl + 1);
    double *h;
    { /* lowpass, cutoff 1/max(up,down) of Nyquist, Kaiser(5.0), scaled by `up` */
        double fc = 1.0 / (double)maxr;
        h = xmalloc(nt * sizeof *h);
        double s = 0.0, alpha = 0.5 * (nt - 1);
        for (int i = 0; i < nt; i++)
        {
            double m = i - alpha, r = m / alpha, v = 1.0 - r * r;
            double win = bessel_i0(5.0 * sqrt(v > 0.0 ? v : 0.0)) / bessel_i0(5.0);
            h[i] = fc * sinc(fc * m) * win;
            s += h[i];
        }
        for (int i = 0; i < nt; i++)
            h[i] = h[i] / s * (double)up;
    }
    long long nn = (long long)n * up;
    size_t n_out = (size_t)(nn / down + (nn % down != 0));
    long long pre_pad = down - hl % down, pre_remove = (hl + pre_pad) / down;
    double *y = xmalloc(n_out * sizeof *y);
    for (size_t i = 0; i < n_out; i++)
    {
        long long c = (pre_remove + (long long)i) * down - pre_pad;
        long long lo = c - 2 * hl;
        long long jmin = lo <= 0 ? 0 : (lo + up - 1) / up;
        long long jmax = c / up;
        if (jmax > (long long)n - 1)
            jmax = (long long)n - 1;
        double acc = 0.0;
        for (long long j = jmin; j <= jmax; j++)
            acc += x[j] * h[c - j * up];
        y[i] = acc;
    }
    free(h);
    *nout = n_out;
    return y;
}

/* short-term RMS of a band-limited signal, floored at ENV_FLOOR x global */
static double *envelope(const double *xb, size_t n)
{
    double *S = xmalloc((n + 1) * sizeof *S), *env = xmalloc(n * sizeof *env);
    S[0] = 0.0;
    for (size_t i = 0; i < n; i++)
        S[i + 1] = S[i] + xb[i] * xb[i];
    double floor_ = ENV_FLOOR * (sqrt(S[n] / (double)n) + 1e-12);
    for (size_t i = 0; i < n; i++)
    {
        size_t lo = i >= (ENV_WIN / 2) ? i - ENV_WIN / 2 : 0;
        size_t hi = i + (ENV_WIN - 1) / 2;
        if (hi > n - 1)
            hi = n - 1;
        double v = (S[hi + 1] - S[lo]) / ENV_WIN;
        v = sqrt(v > 0.0 ? v : 0.0);
        env[i] = v > floor_ ? v : floor_;
    }
    free(S);
    return env;
}

/* keyed band-limited noise carrier, unit RMS */
static double *make_carrier(const uint8_t ck[32], size_t n, const double band[2])
{
    uint8_t sk[32];
    subkey(ck, "pn", sk);
    uint8_t *bits = keyed_bits(sk, n);
    double *chips = xmalloc(n * sizeof *chips);
    for (size_t i = 0; i < n; i++)
        chips[i] = 1.0 - 2.0 * bits[i];
    free(bits);
    double *p = bandpass(chips, n, band);
    int nh;
    double *h = bp_fir(band, &nh);
    double e = 0.0;
    for (int i = 0; i < nh; i++)
        e += h[i] * h[i];
    e = sqrt(e);
    for (size_t i = 0; i < n; i++)
        p[i] /= e;
    free(h);
    free(chips);
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  audio I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    int fs;
    int ch;
    size_t n;
    double *d;
} audio_t; /* d[n*ch], interleaved */

static uint8_t *read_file(const char *path, size_t *len)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return NULL;
    size_t cap = 1 << 16, n = 0;
    uint8_t *b = xmalloc(cap);
    for (;;)
    {
        size_t k = fread(b + n, 1, cap - n, f);
        n += k;
        if (n < cap)
            break;
        cap *= 2;
        b = xrealloc(b, cap);
    }
    fclose(f);
    *len = n;
    return b;
}

/* plain RIFF/WAVE: PCM 8/16/24/32 or IEEE float 32/64 -> floats in [-1, 1) */
static int wav_parse(const uint8_t *buf, size_t len, audio_t *a)
{
    if (len < 12 || memcmp(buf, "RIFF", 4) || memcmp(buf + 8, "WAVE", 4))
        return -1;
    size_t pos = 12, data = 0, dsz = 0;
    int tag = 0, ch = 0, bits = 0, have_fmt = 0, have_data = 0;
    uint32_t rate = 0;
    while (pos + 8 <= len)
    {
        uint64_t sz = le32(buf + pos + 4);
        size_t body = pos + 8;
        if (!memcmp(buf + pos, "fmt ", 4))
        {
            if (sz < 16 || body + 16 > len)
                return -1;
            tag = le16(buf + body);
            ch = le16(buf + body + 2);
            rate = le32(buf + body + 4);
            bits = le16(buf + body + 14);
            if (tag == 0xFFFE && sz >= 40 && body + 26 <= len)
                tag = le16(buf + body + 24);
            have_fmt = 1;
        }
        else if (!memcmp(buf + pos, "data", 4))
        {
            if (sz > len - body)
                sz = len - body;
            data = body;
            dsz = (size_t)sz;
            have_data = 1;
            break;
        }
        pos = body + (size_t)sz + (sz & 1);
    }
    if (!have_fmt || !have_data || ch < 1 || rate == 0)
        return -1;
    int bps = bits / 8;
    if (tag == 1)
    {
        if (bits != 8 && bits != 16 && bits != 24 && bits != 32)
            return -1;
    }
    else if (tag == 3)
    {
        if (bits != 32 && bits != 64)
            return -1;
    }
    else
        return -1;
    size_t frames = dsz / ((size_t)ch * bps);
    double *d = xmalloc(frames * ch * sizeof *d);
    const uint8_t *p = buf + data;
    for (size_t i = 0; i < frames * ch; i++, p += bps)
    {
        double v;
        if (tag == 3)
        {
            if (bits == 32)
            {
                uint32_t u = le32(p);
                float f;
                memcpy(&f, &u, 4);
                v = f;
            }
            else
            {
                uint64_t u = (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32;
                memcpy(&v, &u, 8);
            }
        }
        else if (bits == 8)
            v = ((double)p[0] - 128.0) / 128.0;
        else if (bits == 16)
            v = (double)(int16_t)le16(p) / 32768.0;
        else if (bits == 24)
            v = (double)(int32_t)((uint32_t)p[0] << 8 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 24) / 2147483648.0;
        else
            v = (double)(int32_t)le32(p) / 2147483648.0;
        d[i] = v;
    }
    a->fs = (int)rate;
    a->ch = ch;
    a->n = frames;
    a->d = d;
    return 0;
}

/* not a plain PCM WAV -> let ffmpeg convert it */
static int ffmpeg_to_wav(const char *in, audio_t *a)
{
    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !*tmpdir)
        tmpdir = "/tmp";
    char dir[512], out[600];
    snprintf(dir, sizeof dir, "%s/wavygravy-XXXXXX", tmpdir);
    if (!mkdtemp(dir))
        return -1;
    snprintf(out, sizeof out, "%s/in.wav", dir);
    int ok = 0;
    pid_t pid = fork();
    if (pid == 0)
    {
        if (!freopen("/dev/null", "w", stderr))
            _exit(127);
        if (!freopen("/dev/null", "w", stdout))
            _exit(127);
        execlp("ffmpeg", "ffmpeg", "-v", "error", "-y", "-i", in, "-f", "wav", "-acodec", "pcm_s16le", out, (char *)NULL);
        _exit(127);
    }
    else if (pid > 0)
    {
        int st = 0;
        if (waitpid(pid, &st, 0) == pid && WIFEXITED(st) && WEXITSTATUS(st) == 0)
        {
            size_t len;
            uint8_t *b = read_file(out, &len);
            if (b)
            {
                ok = (wav_parse(b, len, a) == 0);
                free(b);
            }
        }
    }
    unlink(out);
    rmdir(dir);
    return ok ? 0 : -1;
}

static int read_audio(const char *path, audio_t *a)
{
    size_t len;
    uint8_t *b = read_file(path, &len);
    if (b)
    {
        int r = wav_parse(b, len, a);
        free(b);
        if (r == 0)
            return 0;
    }
    if (ffmpeg_to_wav(path, a) == 0)
        return 0;
    return fail("cannot read %s (not a PCM WAV, and ffmpeg could not convert it)", path);
}

static int write_audio(const char *path, int fs, int ch, size_t n, const double *x)
{
    size_t total = n * ch;
    double peak = 0.0, scale = 1.0;
    for (size_t i = 0; i < total; i++)
    {
        double v = fabs(x[i]);
        if (v > peak)
            peak = v;
    }
    if (peak > 0.999)
        scale = 0.999 / peak;
    FILE *f = fopen(path, "wb");
    if (!f)
        return fail("cannot write %s: %s", path, strerror(errno));
    uint8_t hdr[44];
    uint32_t dbytes = (uint32_t)(total * 2);
    memcpy(hdr, "RIFF", 4);
    put_le32(hdr + 4, 36 + dbytes);
    memcpy(hdr + 8, "WAVEfmt ", 8);
    put_le32(hdr + 16, 16);
    hdr[20] = 1;
    hdr[21] = 0;
    hdr[22] = (uint8_t)ch;
    hdr[23] = 0;
    put_le32(hdr + 24, (uint32_t)fs);
    put_le32(hdr + 28, (uint32_t)fs * ch * 2);
    hdr[32] = (uint8_t)(ch * 2);
    hdr[33] = 0;
    hdr[34] = 16;
    hdr[35] = 0;
    memcpy(hdr + 36, "data", 4);
    put_le32(hdr + 40, dbytes);
    uint8_t *pcm = xmalloc(total * 2 + 1);
    for (size_t i = 0; i < total; i++)
    {
        double v = nearbyint(x[i] * scale * 32767.0); /* round-half-even, like np.round */
        if (v > 32767.0)
            v = 32767.0;
        if (v < -32768.0)
            v = -32768.0;
        int16_t s = (int16_t)v;
        pcm[2 * i] = (uint8_t)(s & 0xff);
        pcm[2 * i + 1] = (uint8_t)((uint16_t)s >> 8);
    }
    int ok = fwrite(hdr, 1, 44, f) == 44 && fwrite(pcm, 1, total * 2, f) == total * 2;
    ok = (fclose(f) == 0) && ok;
    free(pcm);
    return ok ? 0 : fail("error writing %s", path);
}

static double *to_mono(const audio_t *a)
{
    double *m = xmalloc(a->n * sizeof *m);
    for (size_t i = 0; i < a->n; i++)
    {
        double s = 0.0;
        for (int c = 0; c < a->ch; c++)
            s += a->d[i * a->ch + c];
        m[i] = s / a->ch;
    }
    return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  forward error correction: K=7, r=1/2 convolutional code, soft Viterbi
 * ═══════════════════════════════════════════════════════════════════════════ */

#define G1 0133
#define G2 0171

static int parity32(unsigned v)
{
    int p = 0;
    while (v)
    {
        p ^= 1;
        v &= v - 1;
    }
    return p;
}

static int t_pred[2][64], t_inbit[64];
static double t_exp[2][64][2];

static void init_trellis(void)
{
    for (int ns = 0; ns < 64; ns++)
    {
        t_inbit[ns] = ns >> 5;
        for (int b = 0; b < 2; b++)
        {
            t_pred[b][ns] = ((ns & 31) << 1) | b;
            unsigned reg = ((unsigned)t_inbit[ns] << 6) | (unsigned)t_pred[b][ns];
            t_exp[b][ns][0] = 1 - 2 * parity32(reg & G1);
            t_exp[b][ns][1] = 1 - 2 * parity32(reg & G2);
        }
    }
}

/* bits: one per byte. Appends 6 tail zeros. Returns 2*(nbits+6) coded bits. */
static uint8_t *conv_encode(const uint8_t *bits, size_t nbits)
{
    size_t T = nbits + 6;
    uint8_t *out = xmalloc(2 * T);
    unsigned state = 0;
    for (size_t t = 0; t < T; t++)
    {
        unsigned u = t < nbits ? bits[t] : 0;
        unsigned reg = (u << 6) | state;
        out[2 * t] = (uint8_t)parity32(reg & G1);
        out[2 * t + 1] = (uint8_t)parity32(reg & G2);
        state = reg >> 1;
    }
    return out;
}

/* soft: + means bit 0, - means bit 1, 0 = erasure. Returns info bits (tail
 * removed), one per byte; *nbits receives their count. */
static uint8_t *viterbi_decode(const double *soft, size_t len, size_t *nbits)
{
    size_t T = len / 2;
    double pm[64], npm[64];
    uint8_t *dec = xmalloc(T * 64 + 1);
    for (int i = 0; i < 64; i++)
        pm[i] = -1e18;
    pm[0] = 0.0;
    for (size_t t = 0; t < T; t++)
    {
        double s1 = soft[2 * t], s2 = soft[2 * t + 1];
        for (int ns = 0; ns < 64; ns++)
        {
            double m0 = pm[t_pred[0][ns]] + s1 * t_exp[0][ns][0] + s2 * t_exp[0][ns][1];
            double m1 = pm[t_pred[1][ns]] + s1 * t_exp[1][ns][0] + s2 * t_exp[1][ns][1];
            dec[t * 64 + ns] = m1 > m0;
            npm[ns] = m1 > m0 ? m1 : m0;
        }
        memcpy(pm, npm, sizeof pm);
    }
    uint8_t *bits = xmalloc(T + 1);
    int ns = 0;
    for (size_t t = T; t-- > 0;)
    {
        bits[t] = (uint8_t)t_inbit[ns];
        ns = t_pred[dec[t * 64 + ns]][ns];
    }
    free(dec);
    *nbits = T >= 6 ? T - 6 : 0;
    return bits;
}

/* data bytes -> interleaved coded bits (one per byte). *outlen receives the count. */
static uint8_t *fec_encode(const uint8_t *data, size_t nbytes, const uint8_t ck[32], const char *label, size_t *outlen)
{
    size_t nb = 8 * nbytes;
    uint8_t *bits = xmalloc(nb + 1);
    for (size_t i = 0; i < nb; i++)
        bits[i] = (data[i >> 3] >> (7 - (i & 7))) & 1;
    uint8_t *coded = conv_encode(bits, nb);
    size_t nc = 2 * (nb + 6);
    uint32_t *perm = keyed_perm(ck, label, nc);
    uint8_t *out = xmalloc(nc);
    for (size_t i = 0; i < nc; i++)
        out[i] = coded[perm[i]];
    free(bits);
    free(coded);
    free(perm);
    *outlen = nc;
    return out;
}

/* soft bits -> bytes (caller frees). *outbytes receives the count. */
static uint8_t *fec_decode(const double *soft, size_t len, const uint8_t ck[32], const char *label, size_t *outbytes)
{
    uint32_t *perm = keyed_perm(ck, label, len);
    double *deint = xmalloc(len * sizeof *deint + 8);
    for (size_t i = 0; i < len; i++)
        deint[perm[i]] = soft[i];
    size_t nbits;
    uint8_t *bits = viterbi_decode(deint, len, &nbits);
    size_t nby = (nbits + 7) / 8;
    uint8_t *out = xcalloc(nby, 1);
    for (size_t i = 0; i < nbits; i++)
        out[i >> 3] |= (uint8_t)(bits[i] << (7 - (i & 7)));
    free(perm);
    free(deint);
    free(bits);
    *outbytes = nby;
    return out;
}

/* ── framing ──────────────────────────────────────────────────────────────── */

static uint64_t hdr_coded_bits(void) { return 2 * (8 * HDR_BYTES + 6); }
static uint64_t pay_coded_bits(uint64_t n) { return 2 * (8 * (n + TAG_BYTES) + 6); }

/* number of BPSK symbols (data + sync) for an n-byte message */
static uint64_t plan_positions(uint64_t n)
{
    uint64_t data = hdr_coded_bits() + pay_coded_bits(n);
    uint64_t blocks = (data + DATA_BITS - 1) / DATA_BITS;
    return data + SYNC_BITS * blocks;
}

/* Largest n such that plan_positions(n) * L <= n_samples_8k (or -1). */
static int64_t capacity_bytes(uint64_t n_samples_8k, uint64_t L)
{
#define FITS(n) (plan_positions(n) * L <= n_samples_8k)
    if (!FITS(0))
        return -1;
    uint64_t lo = 0, hi = 1;
    while (FITS(hi))
    {
        lo = hi;
        hi *= 2;
    }
    while (lo < hi)
    {
        uint64_t mid = (lo + hi + 1) / 2;
        if (FITS(mid))
            lo = mid;
        else
            hi = mid - 1;
    }
    return (int64_t)lo;
#undef FITS
}

/* interleave sync words and data into BPSK symbols (+-1) */
static double *build_symbols(const uint8_t *data_bits, size_t nbits, const double sw[SYNC_BITS], size_t *nsym)
{
    size_t blocks = (nbits + DATA_BITS - 1) / DATA_BITS;
    double *sym = xmalloc((nbits + SYNC_BITS * blocks) * sizeof *sym);
    size_t o = 0;
    for (size_t b = 0; b < blocks; b++)
    {
        for (int k = 0; k < SYNC_BITS; k++)
            sym[o++] = sw[k];
        size_t e = (b + 1) * DATA_BITS;
        if (e > nbits)
            e = nbits;
        for (size_t i = b * DATA_BITS; i < e; i++)
            sym[o++] = 1.0 - 2.0 * data_bits[i];
    }
    *nsym = o;
    return sym;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  encoder
 * ═══════════════════════════════════════════════════════════════════════════ */

static int urandom_bytes(uint8_t *p, size_t n)
{
    FILE *f = fopen("/dev/urandom", "rb");
    if (!f)
        return -1;
    size_t k = fread(p, 1, n, f);
    fclose(f);
    return k == n ? 0 : -1;
}

static int encode_audio(const audio_t *cover, const uint8_t *message, size_t n, const char *password,
                        int red, double strength_db, const double band[2], audio_t *out)
{
    if (cover->fs < WORK_FS)
        return fail("cover sample rate %d Hz is below %d Hz", cover->fs, WORK_FS);
    int L = RED_L[red];
    size_t n8;
    double *mono = to_mono(cover);
    double *rs = resample(mono, cover->n, cover->fs, WORK_FS, &n8);
    double *mono8 = bandpass(rs, n8, band);
    free(mono);
    free(rs);

    uint64_t positions = plan_positions(n);
    uint64_t need = positions * (uint64_t)L;
    if (need > n8)
    {
        int64_t cap = capacity_bytes(n8, L);
        free(mono8);
        return fail("message (%zu bytes) needs %.1f s of cover at '%s' redundancy but the cover has %.1f s "
                    "(capacity here: %lld bytes)",
                    n, (double)need / WORK_FS, RED_NAME[red],
                    (double)n8 / WORK_FS, (long long)(cap > 0 ? cap : 0));
    }

    /* frame: [len][salt] in the clear (but FEC'd), then ciphertext+tag */
    uint8_t header[HDR_BYTES], mk[32], ck[32];
    for (int i = 0; i < LEN_BYTES; i++)
        header[i] = (uint8_t)((uint64_t)n >> (8 * (LEN_BYTES - 1 - i)));
    if (urandom_bytes(header + LEN_BYTES, SALT_BYTES) != 0)
    {
        free(mono8);
        return fail("cannot read /dev/urandom");
    }
    msg_key(password, header + LEN_BYTES, mk);
    uint8_t *ct = xmalloc(n + TAG_BYTES);
    aead_encrypt(mk, ZERO_NONCE, header, HDR_BYTES, message, n, ct);
    carrier_key(password, ck);

    size_t nh_bits, np_bits;
    uint8_t *hb = fec_encode(header, HDR_BYTES, ck, "perm-hdr", &nh_bits);
    uint8_t *pb = fec_encode(ct, n + TAG_BYTES, ck, "perm-pay", &np_bits);
    uint8_t *data_bits = xmalloc(nh_bits + np_bits);
    memcpy(data_bits, hb, nh_bits);
    memcpy(data_bits + nh_bits, pb, np_bits);
    double sw[SYNC_BITS];
    sync_word(ck, sw);
    size_t nsym;
    double *sym = build_symbols(data_bits, nh_bits + np_bits, sw, &nsym);
    if (nsym != positions)
    {
        free(mono8);
        return fail("internal error: symbol count mismatch");
    }
    free(hb);
    free(pb);
    free(data_bits);
    free(ct);

    /* spread, shape, add */
    size_t nneed = (size_t)need;
    double *P = make_carrier(ck, nneed, band);
    double *env = envelope(mono8, n8);
    double g = pow(10.0, -strength_db / 20.0);
    double *w8 = xmalloc(nneed * sizeof *w8);
    for (size_t i = 0; i < nneed; i++)
        w8[i] = g * env[i] * sym[i / L] * P[i];
    free(P);
    free(env);
    free(sym);
    free(mono8);

    size_t nw;
    double *w = resample(w8, nneed, WORK_FS, cover->fs, &nw);
    free(w8);

    out->fs = cover->fs;
    out->ch = cover->ch;
    out->n = cover->n;
    out->d = xmalloc(cover->n * cover->ch * sizeof *out->d);
    for (size_t i = 0; i < cover->n; i++)
    {
        double wi = i < nw ? w[i] : 0.0;
        for (int c = 0; c < cover->ch; c++)
            out->d[i * cover->ch + c] = cover->d[i * cover->ch + c] + wi;
    }
    free(w);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  decoder
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
    size_t b;
    double s;
} anchor_t;

/* Block-by-block demodulator: sync re-lock (timing/drift), I/Q despreading,
 * and a second-order decision-directed phase/frequency tracking loop. */
typedef struct
{
    const double *xs;
    size_t N;
    const cx *pa;
    const double *sw;
    long L;
    double t0;
    size_t b;
    anchor_t *anch;
    size_t na, acap;
    double eps, theta, omega;
    double *soft;
    size_t ns, scap;
} demod_t;

static int dm_z(const demod_t *d, long long start, size_t p, cx *z)
{
    long long L = d->L;
    if (start < 0 || start + L > (long long)d->N || (long long)((p + 1) * (size_t)L) > (long long)d->N)
        return 0;
    const double *x = d->xs + start;
    const cx *q = d->pa + p * (size_t)L;
    double re = 0.0, im = 0.0;
    for (long long i = 0; i < L; i++)
    {
        re += x[i] * creal(q[i]);
        im += x[i] * cimag(q[i]);
    }
    *z = re + im * I;
    return 1;
}

static double dm_sync_metric(const demod_t *d, double tau, size_t b)
{
    double tot = 0.0;
    for (int k = 0; k < SYNC_BITS; k++)
    {
        cx z;
        long long st = (long long)nearbyint(tau + (double)(k * d->L) * (1.0 + d->eps));
        if (!dm_z(d, st, b * BLOCK + k, &z))
            return -1.0;
        tot += creal(z) * creal(z) + cimag(z) * cimag(z);
    }
    return tot;
}

static double dm_relock(demod_t *d, size_t b)
{
    enum
    {
        W = 4
    };
    double pred;
    if (b == 0)
        pred = d->t0;
    else
    {
        const anchor_t *la = &d->anch[d->na - 1];
        pred = la->s + (double)((b - la->b) * BLOCK * (size_t)d->L) * (1.0 + d->eps);
    }
    long long c = (long long)nearbyint(pred);
    double m[2 * W + 1], taus[2 * W + 1];
    int ntaus = 2 * W + 1, bi = 0;
    for (int i = 0; i < ntaus; i++)
    {
        taus[i] = (double)(c - W + i);
        m[i] = dm_sync_metric(d, taus[i], b);
        if (m[i] > m[bi])
            bi = i; /* first max, like argmax */
    }
    double s = taus[bi];
    if (bi > 0 && bi < ntaus - 1 && m[bi] > 0)
    { /* parabolic sub-sample refinement */
        double dd = m[bi - 1] - 2 * m[bi] + m[bi + 1];
        if (dd < 0)
            s += 0.5 * (m[bi - 1] - m[bi + 1]) / dd;
    }
    if (d->na == d->acap)
    {
        d->acap = d->acap ? d->acap * 2 : 64;
        d->anch = xrealloc(d->anch, d->acap * sizeof *d->anch);
    }
    d->anch[d->na].b = b;
    d->anch[d->na].s = s;
    d->na++;
    if (d->na >= 2)
    { /* least-squares slope of anchor position vs. nominal */
        double mx = 0, my = 0, sxx = 0, sxy = 0;
        for (size_t i = 0; i < d->na; i++)
        {
            mx += (double)(d->anch[i].b * BLOCK * (size_t)d->L);
            my += d->anch[i].s;
        }
        mx /= (double)d->na;
        my /= (double)d->na;
        for (size_t i = 0; i < d->na; i++)
        {
            double x = (double)(d->anch[i].b * BLOCK * (size_t)d->L) - mx, y = d->anch[i].s - my;
            sxx += x * x;
            sxy += x * y;
        }
        double e = sxy / sxx - 1.0;
        d->eps = e < -3e-3 ? -3e-3 : (e > 3e-3 ? 3e-3 : e);
    }
    return s;
}

static void dm_init_phase(demod_t *d, const cx *zs)
{
    double ang[SYNC_BITS], w[SYNC_BITS];
    for (int k = 0; k < SYNC_BITS; k++)
    {
        cx u = zs[k] * d->sw[k];
        ang[k] = atan2(cimag(u), creal(u));
        w[k] = cabs(u) + 1e-12;
    }
    /* np.unwrap */
    double corr = 0.0, prev = ang[0];
    for (int k = 1; k < SYNC_BITS; k++)
    {
        double diff = ang[k] - prev, r = fmod(diff + PI, 2 * PI);
        if (r < 0)
            r += 2 * PI;
        double ddm = r - PI;
        if (ddm == -PI && diff > 0)
            ddm = PI;
        double pc = ddm - diff;
        if (fabs(diff) < PI)
            pc = 0.0;
        prev = ang[k];
        corr += pc;
        ang[k] += corr;
    }
    /* weighted LS line fit (np.polyfit(..., w=w) minimises sum (w*residual)^2) */
    double sw_ = 0, sx = 0, sy = 0;
    for (int k = 0; k < SYNC_BITS; k++)
    {
        double W2 = w[k] * w[k];
        sw_ += W2;
        sx += W2 * k;
        sy += W2 * ang[k];
    }
    double xm = sx / sw_, ym = sy / sw_, sxx = 0, sxy = 0;
    for (int k = 0; k < SYNC_BITS; k++)
    {
        double W2 = w[k] * w[k];
        sxx += W2 * (k - xm) * (k - xm);
        sxy += W2 * (k - xm) * (ang[k] - ym);
    }
    double slope = sxy / sxx, icpt = ym - slope * xm;
    d->omega = slope;
    d->theta = icpt - slope;
}

static void dm_block(demod_t *d)
{
    size_t b = d->b;
    long L = d->L;
    double s = dm_relock(d, b);
    cx zs[BLOCK];
    int okz[BLOCK];
    for (int k = 0; k < BLOCK; k++)
    {
        long long st = (long long)nearbyint(s + (double)(k * L) * (1.0 + d->eps));
        okz[k] = dm_z(d, st, b * BLOCK + k, &zs[k]);
    }
    if (b == 0)
    {
        int all = 1;
        for (int k = 0; k < SYNC_BITS; k++)
            all &= okz[k];
        if (all)
            dm_init_phase(d, zs);
    }
    const double alpha = 0.10, beta = 0.003;
    for (int k = 0; k < BLOCK; k++)
    {
        if (!okz[k])
        {
            if (k >= SYNC_BITS)
            {
                if (d->ns == d->scap)
                {
                    d->scap = d->scap ? d->scap * 2 : 4096;
                    d->soft = xrealloc(d->soft, d->scap * sizeof *d->soft);
                }
                d->soft[d->ns++] = 0.0;
            }
            continue;
        }
        d->theta += d->omega;
        cx u = zs[k] * (cos(d->theta) - sin(d->theta) * I); /* z * exp(-j theta) */
        double e;
        if (k < SYNC_BITS)
        {
            cx v = u * d->sw[k];
            e = atan2(cimag(v), creal(v));
        }
        else
        {
            e = atan2(cimag(u) * (creal(u) >= 0 ? 1.0 : -1.0), fabs(creal(u)));
            if (d->ns == d->scap)
            {
                d->scap = d->scap ? d->scap * 2 : 4096;
                d->soft = xrealloc(d->soft, d->scap * sizeof *d->soft);
            }
            d->soft[d->ns++] = creal(u);
        }
        d->theta += alpha * e;
        d->omega += beta * e;
    }
    d->b++;
}

/* returns nbits soft values (zero-padded if the audio runs out); caller frees */
static double *dm_get(demod_t *d, size_t nbits)
{
    while (d->ns < nbits)
    {
        if (d->b * BLOCK * (size_t)d->L + (size_t)BLOCK * d->L / 2 > d->N)
            break;
        dm_block(d);
    }
    double *s = xcalloc(nbits, sizeof *s);
    memcpy(s, d->soft, (d->ns < nbits ? d->ns : nbits) * sizeof *s);
    return s;
}

/* one decode attempt at a given sync candidate; returns 0 + message, or -1 */
static int try_decode(const double *xs, size_t N, const cx *pa, const double *sw, int L, long lag,
                      const uint8_t ck[32], const char *password, int verbose,
                      uint8_t **msg, size_t *msglen)
{
    demod_t d;
    memset(&d, 0, sizeof d);
    d.xs = xs;
    d.N = N;
    d.pa = pa;
    d.sw = sw;
    d.L = L;
    d.t0 = (double)lag;
    int rc = -1;
    uint8_t *hdr = NULL, *ct = NULL;
    double *hs = NULL, *ps = NULL;

    size_t hb = (size_t)hdr_coded_bits(), nh;
    hs = dm_get(&d, hb);
    hdr = fec_decode(hs, hb, ck, "perm-hdr", &nh);
    uint64_t n = 0;
    for (int i = 0; i < LEN_BYTES; i++)
        n = (n << 8) | hdr[i];
    const uint8_t *salt = hdr + LEN_BYTES;
    /* A wrong sync candidate can decode the header to a garbage length; bound
     * it by what could still fit in the remaining audio and bail out early. */
    uint64_t max_bits = ((N > (size_t)lag ? N - lag : 0) / (uint64_t)L + 1) * DATA_BITS;
    if (n > (uint64_t)N || pay_coded_bits(n) > max_bits)
        goto done;
    size_t pb = (size_t)pay_coded_bits(n);
    ps = dm_get(&d, hb + pb);
    size_t nct;
    ct = fec_decode(ps + hb, pb, ck, "perm-pay", &nct);
    if (verbose && d.na)
        fprintf(stderr, "    lag=%ld len=%llu drift=%+.0f ppm blocks=%zu\n", lag, (unsigned long long)n, d.eps * 1e6, d.b);
    {
        uint8_t mk[32];
        msg_key(password, salt, mk);
        uint8_t *pt = xmalloc((size_t)n + 1);
        if (nct == (size_t)n + TAG_BYTES && aead_decrypt(mk, ZERO_NONCE, hdr, HDR_BYTES, ct, (size_t)n, pt) == 0)
        {
            *msg = pt;
            *msglen = (size_t)n;
            rc = 0;
        }
        else
            free(pt);
    }
done:
    free(hdr);
    free(ct);
    free(hs);
    free(ps);
    free(d.anch);
    free(d.soft);
    return rc;
}

/* ── sync search ── */

typedef int (*cand_cb)(void *ctx, double delta, const double *xs, long lag, double z);

typedef struct
{
    const double *xn;
    size_t N, M, T;
    int L;
    cx *refs[SYNC_BITS];
    cx *X, *tmp;
    double *metric, *xs;
    cx *analytic;
} sync_t;

static void sync_make_xs(sync_t *c, double delta, double *xs)
{
    if (delta == 0)
    {
        memcpy(xs, c->xn, c->N * sizeof *xs);
        return;
    }
    if (!c->analytic)
        c->analytic = hilbert(c->xn, c->N);
    for (size_t n = 0; n < c->N; n++)
    {
        double ph = -2.0 * PI * delta * (double)n / WORK_FS;
        xs[n] = creal(c->analytic[n]) * cos(ph) - cimag(c->analytic[n]) * sin(ph);
    }
}

static void sync_evaluate(sync_t *c, double delta, long *lag, double *z)
{
    sync_make_xs(c, delta, c->xs);
    for (size_t i = 0; i < c->M; i++)
        c->X[i] = i < c->N ? c->xs[i] : 0.0;
    fft_fwd_brev(c->X, c->M); /* bit-reversed spectrum; refs[] use the same order */
    memset(c->metric, 0, c->T * sizeof *c->metric);
    for (int k = 0; k < SYNC_BITS; k++)
    {
        const double *xd = (const double *)c->X, *rd = (const double *)c->refs[k];
        double *td = (double *)c->tmp;
        for (size_t i = 0; i < c->M; i++)
        {
            td[2 * i] = xd[2 * i] * rd[2 * i] - xd[2 * i + 1] * rd[2 * i + 1];
            td[2 * i + 1] = xd[2 * i] * rd[2 * i + 1] + xd[2 * i + 1] * rd[2 * i];
        }
        fft_inv_from_brev(c->tmp, c->M);
        for (size_t i = 0; i < c->T; i++)
        {
            double re = td[2 * (k * (size_t)c->L + i)], im = td[2 * (k * (size_t)c->L + i) + 1];
            c->metric[i] += re * re + im * im;
        }
    }
    size_t bi = 0;
    double mean = 0.0;
    for (size_t i = 0; i < c->T; i++)
    {
        if (c->metric[i] > c->metric[bi])
            bi = i;
        mean += c->metric[i];
    }
    mean /= (double)c->T;
    double var = 0.0;
    for (size_t i = 0; i < c->T; i++)
    {
        double dd = c->metric[i] - mean;
        var += dd * dd;
    }
    var /= (double)c->T;
    *lag = (long)bi;
    *z = (c->metric[bi] - mean) / (sqrt(var) + 1e-30);
}

/* calls cb for each sync candidate (best-first per frequency region); stops
 * when cb returns non-zero (and returns that value). */
static int sync_candidates(const double *xn, size_t N, const cx *pa, int L, double max_freq,
                           cand_cb cb, void *ctx, int verbose)
{
    if (N < (size_t)SYNC_BITS * L * 2)
        return 0;
    sync_t c;
    memset(&c, 0, sizeof c);
    c.xn = xn;
    c.N = N;
    c.L = L;
    c.M = next_pow2(N);
    c.T = N - (size_t)SYNC_BITS * L + 1;
    for (int k = 0; k < SYNC_BITS; k++)
    {
        cx *r = xcalloc(c.M, sizeof *r);
        for (int i = 0; i < L; i++)
            r[i] = conj(pa[(size_t)k * L + i]);
        fft_fwd_brev(r, c.M);
        for (size_t i = 0; i < c.M; i++)
            r[i] = conj(r[i]);
        c.refs[k] = r;
    }
    c.X = xmalloc(c.M * sizeof *c.X);
    c.tmp = xmalloc(c.M * sizeof *c.tmp);
    c.metric = xmalloc(c.T * sizeof *c.metric);
    c.xs = xmalloc(N * sizeof *c.xs);

    int steps = (int)nearbyint(max_freq / FREQ_STEP);
    size_t ngrid = steps >= 0 ? 2 * (size_t)steps + 1 : 0;
    double *grid = xmalloc((ngrid + 1) * sizeof *grid);
    size_t ng = 0;
    if (steps >= 0)
    {
        grid[ng++] = 0.0;
        for (int j = 1; j <= steps; j++)
        {
            grid[ng++] = j * FREQ_STEP;
            grid[ng++] = -j * FREQ_STEP;
        }
    }
    /* cache of evaluated offsets, in insertion order */
    double *cd = xmalloc((ngrid + 1) * sizeof *cd), *cz = xmalloc((ngrid + 1) * sizeof *cz);
    long *cl = xmalloc((ngrid + 1) * sizeof *cl);
    size_t nc = 0;
    int result = 0;
    double *xs_best = xmalloc(N * sizeof *xs_best);

#define IS_DONE(v, out)                       \
    do                                        \
    {                                         \
        double v_ = (v);                      \
        out = 0;                              \
        for (size_t qq_ = 0; qq_ < nc; qq_++) \
            if (fabs(cd[qq_] - v_) < 1e-9)    \
            {                                 \
                out = 1;                      \
                break;                        \
            }                                 \
    } while (0)
    for (size_t gi = 0; gi < ng && !result; gi++)
    {
        double d = grid[gi];
        int done;
        IS_DONE(d, done);
        if (done)
            continue;
        sync_evaluate(&c, d, &cl[nc], &cz[nc]);
        cd[nc] = d;
        nc++;
        if (verbose)
            fprintf(stderr, "  L=%d \xCE\xB4=%+.1f Hz: sync z=%.1f\n", L, d, cz[nc - 1]);
        if (cz[nc - 1] >= SYNC_Z)
        {
            const double nb[4] = {d - 5, d - 2.5, d + 2.5, d + 5};
            for (int q = 0; q < 4; q++)
            {
                int dn;
                IS_DONE(nb[q], dn);
                if (fabs(nb[q]) <= max_freq + 1e-9 && !dn)
                {
                    sync_evaluate(&c, nb[q], &cl[nc], &cz[nc]);
                    cd[nc] = nb[q];
                    nc++;
                }
            }
            size_t best = (size_t)-1;
            for (size_t q = 0; q < nc; q++)
                if (fabs(cd[q] - d) <= 5 + 1e-9 && (best == (size_t)-1 || cz[q] > cz[best]))
                    best = q;
            sync_make_xs(&c, cd[best], xs_best);
            result = cb(ctx, cd[best], xs_best, cl[best], cz[best]);
        }
    }
#undef IS_DONE
    free(xs_best);
    free(grid);
    free(cd);
    free(cz);
    free(cl);
    for (int k = 0; k < SYNC_BITS; k++)
        free(c.refs[k]);
    free(c.X);
    free(c.tmp);
    free(c.metric);
    free(c.xs);
    free(c.analytic);
    return result;
}

typedef struct
{
    size_t N;
    const cx *pa;
    const double *sw;
    int L;
    const uint8_t *ck;
    const char *pw;
    int verbose;
    uint8_t *msg;
    size_t msglen;
} try_ctx;

static int try_cb(void *vctx, double delta, const double *xs, long lag, double z)
{
    try_ctx *t = vctx;
    if (t->verbose)
        fprintf(stderr, "  candidate: L=%d \xCE\xB4=%+.1f Hz lag=%ld z=%.1f\n", t->L, delta, lag, z);
    return try_decode(xs, t->N, t->pa, t->sw, t->L, lag, t->ck, t->pw, t->verbose, &t->msg, &t->msglen) == 0;
}

static int decode_audio(const audio_t *a, const char *password, const double band[2], double max_freq,
                        int verbose, uint8_t **msg, size_t *msglen)
{
    uint8_t ck[32];
    carrier_key(password, ck);
    double sw[SYNC_BITS];
    sync_word(ck, sw);
    size_t n8;
    double *mono = to_mono(a), *rs = resample(mono, a->n, a->fs, WORK_FS, &n8);
    free(mono);
    if (n8 < 2)
    {
        free(rs);
        return fail("no message found (wrong password, wrong --band, or the audio is too degraded)");
    }
    double *r8 = bandpass(rs, n8, band);
    free(rs);
    double *env = envelope(r8, n8);
    for (size_t i = 0; i < n8; i++)
        r8[i] /= env[i]; /* AGC: gain-invariant, flattens cover loudness */
    free(env);
    double *xn = r8;
    double *car = make_carrier(ck, n8, band);
    cx *pa = hilbert(car, n8);
    free(car);

    static const int order[3] = {1, 2, 0}; /* medium, high, low */
    int found = 0;
    for (int oi = 0; oi < 3 && !found; oi++)
    {
        try_ctx t = {n8, pa, sw, RED_L[order[oi]], ck, password, verbose, NULL, 0};
        if (sync_candidates(xn, n8, pa, t.L, max_freq, try_cb, &t, verbose))
        {
            *msg = t.msg;
            *msglen = t.msglen;
            found = 1;
        }
    }
    free(xn);
    free(pa);
    if (found)
        return 0;
    return fail("no message found (wrong password, wrong --band, or the audio is too degraded)");
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  channel simulation (for --simulate)
 * ═══════════════════════════════════════════════════════════════════════════ */

static uint64_t rng_state;
static double rng_uniform(void)
{ /* splitmix64 -> (0,1) */
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return ((double)(z >> 11) + 0.5) / 9007199254740992.0;
}
static double rng_normal(void)
{
    return sqrt(-2.0 * log(rng_uniform())) * cos(2.0 * PI * rng_uniform());
}

/* 300-3400 Hz band limit, 8 kHz, 8-bit mu-law, light noise. Returns 8 kHz mono. */
static double *simulate_telephone(const audio_t *a, double snr_db, uint64_t seed, size_t *n_out)
{
    double *mono = to_mono(a);
    size_t n;
    double *y0 = resample(mono, a->n, a->fs, 8000, &n);
    free(mono);
    double *h = firwin(257, 300.0 / 4000.0, 3400.0 / 4000.0, 0.0);
    double *y = conv_same(y0, n, h, 257);
    free(h);
    free(y0);
    double peak = 0.0;
    for (size_t i = 0; i < n; i++)
        if (fabs(y[i]) > peak)
            peak = fabs(y[i]);
    const double mu = 255.0;
    for (size_t i = 0; i < n; i++)
    {
        double v = y[i] / (peak + 1e-12) * 0.9;
        double sg = (v > 0) - (v < 0);
        double c = sg * log1p(mu * fabs(v)) / log1p(mu);
        c = nearbyint(c * 127.0) / 127.0;
        sg = (c > 0) - (c < 0);
        y[i] = sg * expm1(fabs(c) * log1p(mu)) / mu;
    }
    double ms = 0.0;
    for (size_t i = 0; i < n; i++)
        ms += y[i] * y[i];
    double sigma = sqrt(ms / (double)n) * pow(10.0, -snr_db / 20.0);
    rng_state = seed;
    for (size_t i = 0; i < n; i++)
        y[i] += sigma * rng_normal();
    *n_out = n;
    return y;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  CLI
 * ═══════════════════════════════════════════════════════════════════════════ */

static char *read_password_noecho(const char *prompt)
{
    FILE *tty = fopen("/dev/tty", "r+");
    FILE *in = tty ? tty : stdin, *outp = tty ? tty : stderr;
    struct termios old, nw;
    int have = 0;
    if (tcgetattr(fileno(in), &old) == 0)
    {
        nw = old;
        nw.c_lflag &= ~(tcflag_t)ECHO;
        tcsetattr(fileno(in), TCSAFLUSH, &nw);
        have = 1;
    }
    fputs(prompt, outp);
    fflush(outp);
    char buf[4096];
    char *r = fgets(buf, sizeof buf, in);
    if (have)
    {
        tcsetattr(fileno(in), TCSAFLUSH, &old);
        fputc('\n', outp);
    }
    if (tty)
        fclose(tty);
    if (!r)
        buf[0] = 0;
    buf[strcspn(buf, "\r\n")] = 0;
    return strdup(buf);
}

static char *get_password(int confirm)
{
    const char *env = getenv("WAVYGRAVY_PASSWORD");
    if (env && *env)
        return strdup(env);
    char *pw = read_password_noecho("password: ");
    if (confirm)
    {
        char *again = read_password_noecho("again:    ");
        int same = strcmp(again, pw) == 0;
        free(again);
        if (!same)
        {
            free(pw);
            fail("passwords do not match");
            return NULL;
        }
    }
    if (!*pw)
    {
        free(pw);
        fail("empty password");
        return NULL;
    }
    return pw;
}

static int parse_band(const char *s, double band[2])
{
    const char *dash = strchr(s, '-');
    if (dash && !strchr(dash + 1, '-'))
    {
        char *e1, *e2;
        double lo = strtod(s, &e1), hi = strtod(dash + 1, &e2);
        if (e1 == dash && dash[1] && *e2 == 0 && lo > 0 && lo < hi && hi < WORK_FS / 2.0)
        {
            band[0] = lo;
            band[1] = hi;
            return 0;
        }
    }
    return fail("--band must look like 300-2700");
}

static void usage(FILE *f)
{
    fprintf(f,
            "usage: wavygravy {encode,decode,capacity} ...\n\n"
            "  encode   <cover.wav> <message|file> <output.wav>\n"
            "           [--redundancy low|medium|high] [--strength dB]\n"
            "           [--simulate telephone] [--band 300-2700] [-v]\n"
            "  decode   <input.wav> [output.txt] [--max-freq-offset Hz] [--band 300-2700] [-v]\n"
            "  capacity <cover.wav>\n\n"
            "Password: $WAVYGRAVY_PASSWORD, or prompted for (not echoed).\n");
}

static int is_regular_file(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
}

int main(int argc, char **argv)
{
    init_trellis();
    if (argc < 2)
    {
        usage(stderr);
        return 2;
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))
    {
        usage(stdout);
        return 0;
    }
    const char *cmd = argv[1];
    int is_enc = !strcmp(cmd, "encode"), is_dec = !strcmp(cmd, "decode"), is_cap = !strcmp(cmd, "capacity");
    if (!is_enc && !is_dec && !is_cap)
    {
        fprintf(stderr, "wavygravy: invalid command '%s'\n", cmd);
        usage(stderr);
        return 2;
    }

    const char *pos[3] = {NULL, NULL, NULL};
    int npos = 0;
    const char *band_s = "300-2700", *red_s = "medium", *sim_s = NULL;
    double strength = DEFAULT_STRENGTH_DB, max_freq = 30.0;
    int verbose = 0;
    for (int i = 2; i < argc; i++)
    {
        char *arg = argv[i], *val = NULL;
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help"))
        {
            usage(stdout);
            return 0;
        }
        if (!strcmp(arg, "-v") || !strcmp(arg, "--verbose"))
        {
            verbose = 1;
            continue;
        }
        if (arg[0] == '-' && arg[1] == '-')
        {
            char *eq = strchr(arg, '=');
            if (eq)
            {
                *eq = 0;
                val = eq + 1;
            }
            const char *name = arg;
#define NEEDVAL                                                         \
    do                                                                  \
    {                                                                   \
        if (!val)                                                       \
        {                                                               \
            if (i + 1 >= argc)                                          \
            {                                                           \
                fprintf(stderr, "wavygravy: %s needs a value\n", name); \
                return 2;                                               \
            }                                                           \
            val = argv[++i];                                            \
        }                                                               \
    } while (0)
            if (!strcmp(name, "--band"))
            {
                NEEDVAL;
                band_s = val;
            }
            else if (is_enc && !strcmp(name, "--redundancy"))
            {
                NEEDVAL;
                red_s = val;
            }
            else if (is_enc && !strcmp(name, "--strength"))
            {
                NEEDVAL;
                strength = atof(val);
            }
            else if (is_enc && !strcmp(name, "--simulate"))
            {
                NEEDVAL;
                sim_s = val;
            }
            else if (is_dec && !strcmp(name, "--max-freq-offset"))
            {
                NEEDVAL;
                max_freq = atof(val);
            }
            else
            {
                fprintf(stderr, "wavygravy: unrecognized argument '%s'\n", argv[i]);
                return 2;
            }
#undef NEEDVAL
            continue;
        }
        if (npos >= (is_enc ? 3 : is_dec ? 2
                                         : 1))
        {
            fprintf(stderr, "wavygravy: unrecognized argument '%s'\n", arg);
            return 2;
        }
        pos[npos++] = arg;
    }
    if (npos < (is_enc ? 3 : 1))
    {
        fprintf(stderr, "wavygravy: missing arguments\n");
        usage(stderr);
        return 2;
    }

    int red = -1;
    for (int i = 0; i < 3; i++)
        if (!strcmp(red_s, RED_NAME[i]))
            red = i;
    if (red < 0)
    {
        fprintf(stderr, "wavygravy: --redundancy must be low, medium or high\n");
        return 2;
    }
    if (sim_s && strcmp(sim_s, "telephone"))
    {
        fprintf(stderr, "wavygravy: --simulate must be 'telephone'\n");
        return 2;
    }

    audio_t a = {0};

    if (is_cap)
    {
        if (read_audio(pos[0], &a) != 0)
            goto err;
        printf("%s: %.1f s @ %d Hz\n", pos[0], (double)a.n / a.fs, a.fs);
        uint64_t n8 = (uint64_t)((double)a.n * WORK_FS / a.fs);
        for (int i = 0; i < 3; i++)
        {
            int64_t cap = capacity_bytes(n8, RED_L[i]);
            printf("  %-6s (%4d samples/bit, %5.1f bit/s raw): %lld bytes\n",
                   RED_NAME[i], RED_L[i], (double)WORK_FS / RED_L[i], (long long)(cap > 0 ? cap : 0));
        }
        return 0;
    }

    double band[2];
    if (parse_band(band_s, band) != 0)
        goto err;

    if (is_enc)
    {
        if (read_audio(pos[0], &a) != 0)
            goto err;
        uint8_t *msg;
        size_t mlen;
        if (is_regular_file(pos[1]))
        {
            msg = read_file(pos[1], &mlen);
            if (!msg)
            {
                fail("cannot read %s", pos[1]);
                goto err;
            }
        }
        else
        {
            mlen = strlen(pos[1]);
            msg = (uint8_t *)strdup(pos[1]);
        }
        int confirm = isatty(0) && !getenv("WAVYGRAVY_PASSWORD");
        char *pw = get_password(confirm);
        if (!pw)
            goto err;
        audio_t out = {0};
        if (encode_audio(&a, msg, mlen, pw, red, strength, band, &out) != 0)
            goto err;
        if (write_audio(pos[2], out.fs, out.ch, out.n, out.d) != 0)
            goto err;
        fprintf(stderr, "encoded %zu bytes \xE2\x86\x92 %s\n", mlen, pos[2]);
        if (sim_s)
        {
            audio_t back;
            if (read_audio(pos[2], &back) != 0)
                goto err;
            size_t ny;
            double *y = simulate_telephone(&back, 30.0, 0, &ny);
            audio_t ya = {8000, 1, ny, y};
            uint8_t *dm;
            size_t dl;
            int ok = decode_audio(&ya, pw, band, 30.0, 0, &dm, &dl) == 0 && dl == mlen && memcmp(dm, msg, mlen) == 0;
            fprintf(stderr, "simulate %s: %s\n", sim_s, ok ? "OK" : "FAILED");
            return ok ? 0 : 3;
        }
        return 0;
    }

    /* decode */
    if (read_audio(pos[0], &a) != 0)
        goto err;
    char *pw = get_password(0);
    if (!pw)
        goto err;
    uint8_t *msg;
    size_t mlen;
    if (decode_audio(&a, pw, band, max_freq, verbose, &msg, &mlen) != 0)
        goto err;
    if (pos[1])
    {
        FILE *f = fopen(pos[1], "wb");
        if (!f)
        {
            fail("cannot write %s: %s", pos[1], strerror(errno));
            goto err;
        }
        fwrite(msg, 1, mlen, f);
        fclose(f);
    }
    else
    {
        fwrite(msg, 1, mlen, stdout);
        fflush(stdout);
    }
    return 0;

err:
    fprintf(stderr, "wavygravy: %s\n", g_err);
    return 1;
}