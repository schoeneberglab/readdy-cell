/**
 * Blosc for HDF5 - An HDF5 filter that uses the Blosc compressor.
 *
 * Copyright (C) 2009-2015 Francesc Alted <francesc@blosc.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
    Copyright (C) 2010-2016  Francesc Alted
    http://blosc.org
    License: MIT (see LICENSE.txt)

    Filter program that allows the use of the Blosc filter in HDF5.

    This is based on the LZF filter interface (http://h5py.alfven.org)
    by Andrew Collette.

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include "hdf5.h"
#include "blosc_filter.h"

#if defined(__GNUC__)
#define PUSH_ERR(func, minor, str, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, str, ##__VA_ARGS__)
#elif defined(_MSC_VER)
#define PUSH_ERR(func, minor, str, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, str, __VA_ARGS__)
#else
#define PUSH_ERR(func, minor, ...) H5Epush(H5E_DEFAULT, __FILE__, func, __LINE__, H5E_ERR_CLS, H5E_PLINE, minor, __VA_ARGS__)
#endif

#define GET_FILTER(a, b, c, d, e, f, g) H5Pget_filter_by_id(a,b,c,d,e,f,g,NULL)

size_t blosc_filter(unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t* buf_size, void** buf);

herr_t blosc_set_local(hid_t dcpl, hid_t type, hid_t space);

/* Register the filter, passing on the HDF5 return value */
int register_blosc(char **version, char **date){
    int retval;

    H5Z_class_t filter_class = {
            H5Z_CLASS_T_VERS,
            (H5Z_filter_t)(FILTER_BLOSC),
            1, 1,
            "blosc",
            NULL,
            (H5Z_set_local_func_t)(blosc_set_local),
            (H5Z_func_t)(blosc_filter)
    };

    retval = H5Zregister(&filter_class);
    if(retval<0){
        PUSH_ERR("register_blosc", H5E_CANTREGISTER, "Can't register Blosc filter");
    }
    if (version != NULL && date != NULL) {
        *version = strdup(BLOSC_VERSION_STRING);
        *date = strdup(BLOSC_VERSION_DATE);
    }
    return 1; /* lib is available */
}

/* Filter setup. */
herr_t blosc_set_local(hid_t dcpl, hid_t type, hid_t space) {
    int ndims;
    int i;
    herr_t r;

    unsigned int typesize, basetypesize;
    unsigned int bufsize;
    hsize_t chunkdims[32];
    unsigned int flags;
    size_t nelements = 8;
    unsigned int values[] = {0, 0, 0, 0, 0, 0, 0, 0};
    hid_t super_type;
    H5T_class_t classt;

    r = GET_FILTER(dcpl, FILTER_BLOSC, &flags, &nelements, values, 0, NULL);
    if (r < 0) return -1;

    if (nelements < 4) nelements = 4;

    values[0] = FILTER_BLOSC_VERSION;
    values[1] = BLOSC_VERSION_FORMAT;

    ndims = H5Pget_chunk(dcpl, 32, chunkdims);
    if (ndims < 0) return -1;
    if (ndims > 32) {
        PUSH_ERR("blosc_set_local", H5E_CALLBACK, "Chunk rank exceeds limit");
        return -1;
    }

    typesize = H5Tget_size(type);
    if (typesize == 0) return -1;
    classt = H5Tget_class(type);
    if (classt == H5T_ARRAY) {
        super_type = H5Tget_super(type);
        basetypesize = H5Tget_size(super_type);
        H5Tclose(super_type);
    } else {
        basetypesize = typesize;
    }

    if (basetypesize > BLOSC_MAX_TYPESIZE) basetypesize = 1;
    values[2] = basetypesize;

    bufsize = typesize;
    for (i = 0; i < ndims; i++) {
        bufsize *= chunkdims[i];
    }
    values[3] = bufsize;

#ifdef BLOSC_DEBUG
    fprintf(stderr, "Blosc: Computed buffer size %d\n", bufsize);
#endif

    r = H5Pmodify_filter(dcpl, FILTER_BLOSC, flags, nelements, values);
    if (r < 0) return -1;

    return 1;
}

size_t blosc_filter(unsigned flags, size_t cd_nelmts,
                    const unsigned cd_values[], size_t nbytes,
                    size_t* buf_size, void** buf) {

    void* outbuf = NULL;
    int status = 0;
    size_t typesize;
    size_t outbuf_size;
    int clevel = 5;
    int doshuffle = 1;
    int compcode;
    int code;
    const char* compname = "blosclz";
    const char* complist;
    char errmsg[256];

    typesize = cd_values[2];
    outbuf_size = cd_values[3];
    if (cd_nelmts >= 5) {
        clevel = cd_values[4];
    }
    if (cd_nelmts >= 6) {
        doshuffle = cd_values[5];
#if ((BLOSC_VERSION_MAJOR <= 1) && (BLOSC_VERSION_MINOR < 8))
        if (doshuffle == BLOSC_BITSHUFFLE) {
            PUSH_ERR("blosc_filter", H5E_CALLBACK,
                     "this Blosc library version is not supported.  Please update to >= 1.8");
            goto failed;
        }
#endif
    }
    if (cd_nelmts >= 7) {
        compcode = cd_values[6];
        complist = blosc_list_compressors();
        code = blosc_compcode_to_compname(compcode, &compname);
        if (code == -1) {
            PUSH_ERR("blosc_filter", H5E_CALLBACK,
                     "this Blosc library does not have support for "
                     "the '%s' compressor, but only for: %s",
                     compname, complist);
            goto failed;
        }
    }

    if (!(flags & H5Z_FLAG_REVERSE)) {
        outbuf_size = (*buf_size);

#ifdef BLOSC_DEBUG
        fprintf(stderr, "Blosc: Compress %zd chunk w/buffer %zd\n",
                nbytes, outbuf_size);
#endif

        outbuf = malloc(outbuf_size);

        if (outbuf == NULL) {
            PUSH_ERR("blosc_filter", H5E_CALLBACK,
                     "Can't allocate compression buffer");
            goto failed;
        }

        blosc_set_compressor(compname);
        status = blosc_compress(clevel, doshuffle, typesize, nbytes,
                                *buf, outbuf, nbytes);
        if (status < 0) {
            PUSH_ERR("blosc_filter", H5E_CALLBACK, "Blosc compression error");
            goto failed;
        }

    } else {
        size_t cbytes, blocksize;

        free(outbuf);

        blosc_cbuffer_sizes(*buf, &outbuf_size, &cbytes, &blocksize);

#ifdef BLOSC_DEBUG
        fprintf(stderr, "Blosc: Decompress %zd chunk w/buffer %zd\n", nbytes, outbuf_size);
#endif

        outbuf = malloc(outbuf_size);

        if (outbuf == NULL) {
            PUSH_ERR("blosc_filter", H5E_CALLBACK, "Can't allocate decompression buffer");
            goto failed;
        }

        status = blosc_decompress(*buf, outbuf, outbuf_size);
        if (status <= 0) {
            PUSH_ERR("blosc_filter", H5E_CALLBACK, "Blosc decompression error");
            goto failed;
        }
    }

    if (status != 0) {
        free(*buf);
        *buf = outbuf;
        *buf_size = outbuf_size;
        return status;
    }

failed:
    free(outbuf);
    return 0;
}