/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

struct bmpfile_magic {
    unsigned char magic[2];
};

struct bmpfile_header {
    uint32_t filesz;
    uint16_t creator1;
    uint16_t creator2;
    uint32_t bmp_offset;
};

struct bmpinfo_header {
    uint32_t header_sz;
    int32_t width;
    int32_t height;
    uint16_t nplanes;
    uint16_t bitspp;
    uint32_t compress_type;
    uint32_t bmp_bytesz;
    int32_t hres;
    int32_t vres;
    uint32_t ncolors;
    uint32_t nimpcolors;
};

void printarr(int iter, double *array, int size, int bx, int by, int (*ind) (int, int, int))
{
    int xcnt, ycnt, my_ycnt;
    int i;
    int linesize = bx * 3;
    char *myline;

    char fname[128];
    snprintf(fname, 128, "./output-%i.bmp", iter);

    FILE *fp = fopen(fname, "w");

    struct bmpfile_magic magic;
    struct bmpfile_header header;
    struct bmpinfo_header bmpinfo;

    magic.magic[0] = 0x42;
    magic.magic[1] = 0x4D;

    fwrite(&magic, sizeof(struct bmpfile_magic), 1, fp);

    header.filesz =
        sizeof(struct bmpfile_magic) + sizeof(struct bmpfile_header) +
        sizeof(struct bmpinfo_header) + size * (size * 3 + size * 3 % 4);
    header.creator1 = 0xFE;
    header.creator1 = 0xFE;
    header.bmp_offset =
        sizeof(struct bmpfile_magic) + sizeof(struct bmpfile_header) +
        sizeof(struct bmpinfo_header);

    fwrite(&header, sizeof(struct bmpfile_header), 1, fp);

    bmpinfo.header_sz = sizeof(struct bmpinfo_header);
    bmpinfo.width = size;
    bmpinfo.height = size;
    bmpinfo.nplanes = 1;
    bmpinfo.bitspp = 24;
    bmpinfo.compress_type = 0;
    bmpinfo.bmp_bytesz = size * (size * 3 + size * 3 % 4);
    bmpinfo.hres = size;
    bmpinfo.vres = size;
    bmpinfo.ncolors = 0;
    bmpinfo.nimpcolors = 0;

    fwrite(&bmpinfo, sizeof(struct bmpinfo_header), 1, fp);

    myline = (char *) malloc(linesize);

    my_ycnt = 0;
    xcnt = 0;
    ycnt = size;

    while (ycnt >= 0) {
        for (i = 0; i < linesize; i += 3) {
            int rgb;
            if (i / 3 > bx)
                rgb = 0;
            else
                rgb = (int) round(255.0 * array[ind(i / 3, by - my_ycnt, bx)]);
            if ((i == 0) || (i / 3 == bx - 1) || (my_ycnt == 0) || (my_ycnt == by - 1))
                rgb = 255;
            if (rgb > 255)
                rgb = 255;
            myline[i + 0] = 0;
            myline[i + 1] = 0;
            myline[i + 2] = rgb;
        }
        my_ycnt++;
        fwrite(myline, linesize, 1, fp);
        xcnt += bx;
        if (xcnt >= size) {
            xcnt = 0;
            ycnt--;
        }
    }

    free(myline);

    fclose(fp);
}
