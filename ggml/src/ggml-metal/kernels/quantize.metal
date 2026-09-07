#include "common.h"
#include "dequantize.h"
#include "quantize.h"

template<typename T0, typename T1>
kernel void kernel_cpy_t_t(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device T1 * dst_data = (device T1 *) (dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.ne00;) {
        device const T0 * src = (device T0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + i00*args.nb00);
        dst_data[i00] = (T1) src[0];
        break;
    }
}

typedef decltype(kernel_cpy_t_t<float, float>) kernel_cpy_t;

template [[host_name("kernel_cpy_f32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   float>;
template [[host_name("kernel_cpy_f32_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   half>;
template [[host_name("kernel_cpy_f32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<float,   int32_t>;
template [[host_name("kernel_cpy_i32_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, float>;
template [[host_name("kernel_cpy_i32_i32")]]   kernel kernel_cpy_t kernel_cpy_t_t<int32_t, int32_t>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_f32_bf16")]]  kernel kernel_cpy_t kernel_cpy_t_t<float,   bfloat>;
#endif
template [[host_name("kernel_cpy_f16_f32")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    float>;
template [[host_name("kernel_cpy_f16_f16")]]   kernel kernel_cpy_t kernel_cpy_t_t<half,    half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_cpy_bf16_f32")]]  kernel kernel_cpy_t kernel_cpy_t_t<bfloat,  float>;
template [[host_name("kernel_cpy_bf16_bf16")]] kernel kernel_cpy_t kernel_cpy_t_t<bfloat,  bfloat>;
#endif

template<short QK,
         typename block_q,
         void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_cpy_f32_q(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n / (args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0) / (args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0) / args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK;

    device block_q * dst_data = (device block_q *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK)*args.nb00);

        quantize_func(src, dst_data[i00]);

        break;
    }
}

typedef decltype(kernel_cpy_f32_q<QK8_0,  block_q8_0,  quantize_q8_0>)  cpy_f_q_t;

template [[host_name("kernel_cpy_f32_q8_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK8_0,  block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_cpy_f32_q1_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK1_0,  block_q1_0,   quantize_q1_0>;
template [[host_name("kernel_cpy_f32_q2_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK2_0,  block_q2_0,   quantize_q2_0>;
template [[host_name("kernel_cpy_f32_q4_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK4_0,  block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_cpy_f32_q4_1")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK4_1,  block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_cpy_f32_q5_0")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK5_0,  block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_cpy_f32_q5_1")]]   kernel cpy_f_q_t kernel_cpy_f32_q<QK5_1,  block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_cpy_f32_iq4_nl")]] kernel cpy_f_q_t kernel_cpy_f32_q<QK4_NL, block_iq4_nl, quantize_iq4_nl>;
template [[host_name("kernel_cpy_f32_tq2_0")]]  kernel cpy_f_q_t kernel_cpy_f32_q<QK_K,   block_tq2_0,  quantize_tq2_0>;

template<typename T4x4, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread T4x4 &)>
kernel void kernel_cpy_q_f32(
        constant ggml_metal_kargs_cpy & args,
        device  const char * src0,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;

    if (i01 >= args.ne01) {
        return;
    }

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;

    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_q * src_data = (device const block_q *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device       T4x4    * dst_data = (device       T4x4    *)(dst  +  i3*args.nb3  +  i2*args.nb2  +  i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        T4x4 temp;
        dequantize_func(src_data + i00/nl, i00%nl, temp);
        dst_data[i00] = temp;

        break;
    }
}

typedef decltype(kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>) cpy_q_f_t;

template [[host_name("kernel_cpy_q1_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q1_0, 8, dequantize_q1_0>;
template [[host_name("kernel_cpy_q2_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q2_0, 4, dequantize_q2_0>;
template [[host_name("kernel_cpy_q4_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_q8_0, 2, dequantize_q8_0>;

template [[host_name("kernel_cpy_tq2_0_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_tq2_0, QK_NL, dequantize_tq2_0>;

template [[host_name("kernel_cpy_q1_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q1_0, 8, dequantize_q1_0>;
template [[host_name("kernel_cpy_q2_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q2_0, 4, dequantize_q2_0>;
template [[host_name("kernel_cpy_q4_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_cpy_q4_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_cpy_q5_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_cpy_q5_1_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_cpy_q8_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_q8_0, 2, dequantize_q8_0>;


template [[host_name("kernel_cpy_tq2_0_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_tq2_0, QK_NL, dequantize_tq2_0>;

template [[host_name("kernel_cpy_tq3_1s_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_tq3_1s, 2, dequantize_tq3_1s>;
template [[host_name("kernel_cpy_tq3_1s_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_tq3_1s, 2, dequantize_tq3_1s>;
template [[host_name("kernel_cpy_tq4_1s_f32")]] kernel cpy_q_f_t kernel_cpy_q_f32<float4x4, block_tq4_1s, 2, dequantize_tq4_1s>;
template [[host_name("kernel_cpy_tq4_1s_f16")]] kernel cpy_q_f_t kernel_cpy_q_f32<half4x4, block_tq4_1s, 2, dequantize_tq4_1s>;


template<typename T>
kernel void kernel_concat(
        constant ggml_metal_kargs_concat & args,
        device  const char * src0,
        device  const char * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = ntg.y == 1 ? tgpig.x : tgpig.x*ntg.y + tpitg.y;

    if (i1 >= args.ne1) {
        return;
    }

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        device const T * x;

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const T *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const T *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device T * y = (device T *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

typedef decltype(kernel_concat<float>) kernel_concat_t;

template [[host_name("kernel_concat_f32")]]  kernel kernel_concat_t kernel_concat<float>;
template [[host_name("kernel_concat_f16")]]  kernel kernel_concat_t kernel_concat<half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_concat_bf16")]] kernel kernel_concat_t kernel_concat<bfloat>;
#endif
template [[host_name("kernel_concat_i8")]]   kernel kernel_concat_t kernel_concat<char>;
template [[host_name("kernel_concat_i16")]]  kernel kernel_concat_t kernel_concat<short>;
template [[host_name("kernel_concat_i32")]]  kernel kernel_concat_t kernel_concat<int>;
template [[host_name("kernel_concat_i64")]]  kernel kernel_concat_t kernel_concat<long>;

template<typename block_q>
kernel void kernel_concat_q(
        constant ggml_metal_kargs_concat & args,
        device  const char * src0,
        device  const char * src1,
        device        char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {

    // note: for quantized types, the args are in units of blocks (nb0 == type_size)
    const int i3 = tgpig.z;
    const int i2 = tgpig.y;
    const int i1 = ntg.y == 1 ? tgpig.x : tgpig.x*ntg.y + tpitg.y;

    if (i1 >= args.ne1) {
        return;
    }

    int o[4] = {0, 0, 0, 0};
    o[args.dim] = args.dim == 0 ? args.ne00 : (args.dim == 1 ? args.ne01 : (args.dim == 2 ? args.ne02 : args.ne03));

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        device const block_q * x;

        if (i0 < args.ne00 && i1 < args.ne01 && i2 < args.ne02 && i3 < args.ne03) {
            x = (device const block_q *)(src0 + (i3       )*args.nb03 + (i2       )*args.nb02 + (i1       )*args.nb01 + (i0       )*args.nb00);
        } else {
            x = (device const block_q *)(src1 + (i3 - o[3])*args.nb13 + (i2 - o[2])*args.nb12 + (i1 - o[1])*args.nb11 + (i0 - o[0])*args.nb10);
        }

        device block_q * y = (device block_q *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

        *y = *x;
    }
}

typedef decltype(kernel_concat_q<block_q4_0>) kernel_concat_q_t;

template [[host_name("kernel_concat_q4_0")]] kernel kernel_concat_q_t kernel_concat_q<block_q4_0>;
template [[host_name("kernel_concat_q4_1")]] kernel kernel_concat_q_t kernel_concat_q<block_q4_1>;
template [[host_name("kernel_concat_q5_0")]] kernel kernel_concat_q_t kernel_concat_q<block_q5_0>;
template [[host_name("kernel_concat_q5_1")]] kernel kernel_concat_q_t kernel_concat_q<block_q5_1>;
template [[host_name("kernel_concat_q8_0")]] kernel kernel_concat_q_t kernel_concat_q<block_q8_0>;

template<typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread float4x4 &)>
kernel void kernel_get_rows_q(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg  [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (device const block_q *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (device      float4x4 *) ((      device char *) dst  + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        float4x4 temp;
        dequantize_func(psrc + ind/nl, ind%nl, temp);
        pdst[ind] = temp;

        break;
    }
}

template<typename T0, typename T>
kernel void kernel_get_rows_f(
        constant ggml_metal_kargs_get_rows & args,
        device const void * src0,
        device const void * src1,
        device       void * dst,
        uint3               tgpig[[threadgroup_position_in_grid]],
        ushort              tiitg[[thread_index_in_threadgroup]],
        ushort3             ntg [[threads_per_threadgroup]]) {
    const int32_t iw0 = tgpig.x/args.ne10;
    const int32_t i10 = tgpig.x%args.ne10;
    const int32_t i11 = tgpig.y;
    const int32_t i12 = tgpig.z;

    const int32_t r = ((const device int32_t *) ((const device char *) src1 + i12*args.nb12 + i11*args.nb11 + i10*args.nb10))[0];

    const int32_t i02 = i11;
    const int32_t i03 = i12;

    auto psrc = (const device T0 *) ((const device char *) src0 + i03*args.nb03 + i02*args.nb02 +   r*args.nb01);
    auto pdst = (      device T  *) ((      device char *)  dst + i12*args.nb3  + i11*args.nb2  + i10*args.nb1);

    for (int ind = iw0*ntg.x + tiitg; ind < args.ne00t;) {
        pdst[ind] = psrc[ind];

        break;
    }
}

typedef decltype(kernel_get_rows_f<float, float>) get_rows_f_t;

template [[host_name("kernel_get_rows_f32")]]  kernel get_rows_f_t kernel_get_rows_f<float, float>;
template [[host_name("kernel_get_rows_f16")]]  kernel get_rows_f_t kernel_get_rows_f<half,  float>;
template [[host_name("kernel_get_rows_i32")]]  kernel get_rows_f_t kernel_get_rows_f<int32_t, int32_t>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_get_rows_bf16")]] kernel get_rows_f_t kernel_get_rows_f<bfloat, float>;
#endif

typedef decltype(kernel_get_rows_q<block_q4_0, 2, dequantize_q4_0>) get_rows_q_t;

template [[host_name("kernel_get_rows_q1_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q1_0,    8, dequantize_q1_0>;
template [[host_name("kernel_get_rows_q2_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_0,    4, dequantize_q2_0>;
template [[host_name("kernel_get_rows_q4_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_0,    2, dequantize_q4_0>;
template [[host_name("kernel_get_rows_q4_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_1,    2, dequantize_q4_1>;
template [[host_name("kernel_get_rows_q5_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_0,    2, dequantize_q5_0>;
template [[host_name("kernel_get_rows_q5_1")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_1,    2, dequantize_q5_1>;
template [[host_name("kernel_get_rows_q8_0")]]    kernel get_rows_q_t kernel_get_rows_q<block_q8_0,    2, dequantize_q8_0>;
template [[host_name("kernel_get_rows_mxfp4")]]   kernel get_rows_q_t kernel_get_rows_q<block_mxfp4,   2, dequantize_mxfp4>;
template [[host_name("kernel_get_rows_q2_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q2_K,    QK_NL, dequantize_q2_K>;
template [[host_name("kernel_get_rows_q3_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q3_K,    QK_NL, dequantize_q3_K>;
template [[host_name("kernel_get_rows_q4_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q4_K,    QK_NL, dequantize_q4_K>;
template [[host_name("kernel_get_rows_q5_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q5_K,    QK_NL, dequantize_q5_K>;
template [[host_name("kernel_get_rows_q6_K")]]    kernel get_rows_q_t kernel_get_rows_q<block_q6_K,    QK_NL, dequantize_q6_K>;
template [[host_name("kernel_get_rows_iq2_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq2_xxs, QK_NL, dequantize_iq2_xxs>;
template [[host_name("kernel_get_rows_iq2_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq2_xs,  QK_NL, dequantize_iq2_xs>;
template [[host_name("kernel_get_rows_iq3_xxs")]] kernel get_rows_q_t kernel_get_rows_q<block_iq3_xxs, QK_NL, dequantize_iq3_xxs>;
template [[host_name("kernel_get_rows_iq3_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq3_s,   QK_NL, dequantize_iq3_s>;
template [[host_name("kernel_get_rows_iq2_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq2_s,   QK_NL, dequantize_iq2_s>;
template [[host_name("kernel_get_rows_iq1_s")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_s,   QK_NL, dequantize_iq1_s>;
template [[host_name("kernel_get_rows_iq1_m")]]   kernel get_rows_q_t kernel_get_rows_q<block_iq1_m,   QK_NL, dequantize_iq1_m>;
template [[host_name("kernel_get_rows_iq4_nl")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_nl,  2,     dequantize_iq4_nl>;
template [[host_name("kernel_get_rows_iq4_xs")]]  kernel get_rows_q_t kernel_get_rows_q<block_iq4_xs,  QK_NL, dequantize_iq4_xs>;

template [[host_name("kernel_get_rows_tq2_0")]]   kernel get_rows_q_t kernel_get_rows_q<block_tq2_0,   QK_NL, dequantize_tq2_0>;

template<typename TS, typename TI, short QK, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS      * src_row = (const device TS      *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + QK*ind, dst_row[ind]);
    }
}

template [[host_name("kernel_get_rows_tq3_1s")]]  kernel get_rows_q_t kernel_get_rows_q<block_tq3_1s,  2, dequantize_tq3_1s>;
template [[host_name("kernel_get_rows_tq4_1s")]]  kernel get_rows_q_t kernel_get_rows_q<block_tq4_1s,  2, dequantize_tq4_1s>;


template<typename TS, typename TI, typename block_q, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_q32(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS      * src_row = (const device TS      *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + 32*ind, dst_row[ind]);
    }
}

// TurboQuant set_rows kernel — block size 128 (QK_TURBO3/QK_TURBO4)
// TurboQuant SET_ROWS kernel — processes QK_TURBO3_GROUP (128) elements per iteration,
// writes QK_TURBO3_GROUP/QK_TURBO3 (4) blocks per iteration.
// The rotation operates on 128 elements, then results are split into 32-element blocks.
template<typename TI, typename block_q, int QK, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_turbo(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float   * src_row = (const device float   *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO3_GROUP / QK;  // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO3_GROUP * grp;

        // Normalize and rotate the full 128-element group
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Split into 4 blocks of 32 elements each
        // All blocks store the SAME group norm — centroids are in normalized space
        // Norm correction (ported from @spiritbuun's CUDA implementation):
        // Accumulate ||centroid_vector||^2 across all 128 elements, then store
        // grp_norm / ||centroid_vector|| instead of raw grp_norm. This makes
        // dequantized vectors have the exact original L2 norm at zero decode cost.
        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_q & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK;

            for (int j = 0; j < QK / 4; j++) blk.qs[j] = 0;
            for (int j = 0; j < QK / 8; j++) blk.signs[j] = 0;

            // Quantize rotated values to 3-bit centroids
            for (int j = 0; j < QK; j++) {
                float rv = x[off + j];  // rotated, normalized value
                uint8_t idx;
                if      (rv < turbo_mid_3bit[0]) idx = 0;
                else if (rv < turbo_mid_3bit[1]) idx = 1;
                else if (rv < turbo_mid_3bit[2]) idx = 2;
                else if (rv < turbo_mid_3bit[3]) idx = 3;
                else if (rv < turbo_mid_3bit[4]) idx = 4;
                else if (rv < turbo_mid_3bit[5]) idx = 5;
                else if (rv < turbo_mid_3bit[6]) idx = 6;
                else                              idx = 7;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));

                // Accumulate centroid reconstruction norm for norm correction
                float c = turbo_centroids_3bit[idx];
                recon_norm_sq += c * c;
            }
        }

        // Norm correction: store corrected norm so dequant(x) has exact original L2 norm.
        // Zero decode cost — dequant already multiplies by stored norm.
        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant2 SET_ROWS kernel — 2-bit PolarQuant, 4 centroids, no signs byte.
// Same 128-element group WHT rotation as turbo3, but simpler quantization.
template<typename TI>
kernel void kernel_set_rows_turbo2(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo2_0 * dst_row = (      device block_turbo2_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float           *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2;  // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO2_GROUP * grp;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO2_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_turbo2_0 & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK_TURBO2;

            for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;

            for (int j = 0; j < QK_TURBO2; j++) {
                float rv = x[off + j];
                uint8_t idx;
                if      (rv < turbo_mid_2bit[0]) idx = 0;
                else if (rv < turbo_mid_2bit[1]) idx = 1;
                else if (rv < turbo_mid_2bit[2]) idx = 2;
                else                              idx = 3;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);

                float c = turbo_centroids_2bit[idx];
                recon_norm_sq += c * c;
            }
        }

        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant4 SET_ROWS kernel — processes 128 elements per block.
// Unlike turbo3 (4x32 blocks), turbo4 uses single 128-element blocks
// with packed 3-bit indices and QJL signs.
template<typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo4_0 * dst_row = (      device block_turbo4_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float         *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Each block is one 128-element group (nk0 = ne0 / QK_TURBO4)
    const int n_blocks = args.nk0;

    for (int blk_idx = tiitg%tptg.x; blk_idx < n_blocks; blk_idx += tptg.x) {
        const device float * blk_src = src_row + QK_TURBO4 * blk_idx;
        device block_turbo4_0 & blk = dst_row[blk_idx];

        // Step 1: Compute norm + normalize
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) norm_sq += blk_src[j] * blk_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        float normalized[128];
        for (int j = 0; j < 128; j++) {
            normalized[j] = blk_src[j] * inv_norm;
            x[j] = normalized[j];
        }

        // Step 2: WHT rotate in-place
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Step 3: 4-bit PolarQuant — nibble packing (2 indices per byte)
        for (int j = 0; j < QK_TURBO4 / 2; j++) blk.qs[j] = 0;
        // qs[64] covers full nibble range — no signs field in 4-bit struct

        float recon_norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            float val = x[j];
            uint8_t idx;
            if      (val < turbo_mid_4bit[ 0]) idx = 0;
            else if (val < turbo_mid_4bit[ 1]) idx = 1;
            else if (val < turbo_mid_4bit[ 2]) idx = 2;
            else if (val < turbo_mid_4bit[ 3]) idx = 3;
            else if (val < turbo_mid_4bit[ 4]) idx = 4;
            else if (val < turbo_mid_4bit[ 5]) idx = 5;
            else if (val < turbo_mid_4bit[ 6]) idx = 6;
            else if (val < turbo_mid_4bit[ 7]) idx = 7;
            else if (val < turbo_mid_4bit[ 8]) idx = 8;
            else if (val < turbo_mid_4bit[ 9]) idx = 9;
            else if (val < turbo_mid_4bit[10]) idx = 10;
            else if (val < turbo_mid_4bit[11]) idx = 11;
            else if (val < turbo_mid_4bit[12]) idx = 12;
            else if (val < turbo_mid_4bit[13]) idx = 13;
            else if (val < turbo_mid_4bit[14]) idx = 14;
            else                               idx = 15;

            // 4-bit nibble pack: 2 elements per byte
            blk.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);

            float c = turbo_centroids_4bit[idx];
            recon_norm_sq += c * c;
        }

        // Norm correction
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
    }
}

template<typename TS, typename TI, typename TD>
kernel void kernel_set_rows_f(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device TD * dst_row = (      device TD *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device TS * src_row = (const device TS *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        dst_row[ind] = (TD) src_row[ind];
    }
}

typedef decltype(kernel_set_rows_f<float, int64_t, float>) set_rows_f_t;

template [[host_name("kernel_set_rows_f32_i64_f32")]]   kernel set_rows_f_t kernel_set_rows_f<float, int64_t, float>;
template [[host_name("kernel_set_rows_f32_i32_f32")]]   kernel set_rows_f_t kernel_set_rows_f<float, int32_t, float>;
template [[host_name("kernel_set_rows_f32_i64_f16")]]   kernel set_rows_f_t kernel_set_rows_f<float, int64_t, half>;
template [[host_name("kernel_set_rows_f32_i32_f16")]]   kernel set_rows_f_t kernel_set_rows_f<float, int32_t, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_f32_i64_bf16")]]  kernel set_rows_f_t kernel_set_rows_f<float, int64_t, bfloat>;
template [[host_name("kernel_set_rows_f32_i32_bf16")]]  kernel set_rows_f_t kernel_set_rows_f<float, int32_t, bfloat>;
#endif

template [[host_name("kernel_set_rows_f16_i64_f16")]]   kernel set_rows_f_t kernel_set_rows_f<half, int64_t, half>;
template [[host_name("kernel_set_rows_f16_i32_f16")]]   kernel set_rows_f_t kernel_set_rows_f<half, int32_t, half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_set_rows_bf16_i64_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat, int64_t, bfloat>;
template [[host_name("kernel_set_rows_bf16_i32_bf16")]] kernel set_rows_f_t kernel_set_rows_f<bfloat, int32_t, bfloat>;
#endif

typedef decltype(kernel_set_rows_q32<float, int64_t, block_q8_0, quantize_q8_0>) set_rows_q32_t;

template [[host_name("kernel_set_rows_f32_i64_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_f32_i32_q8_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q8_0,   quantize_q8_0>;
template [[host_name("kernel_set_rows_f32_i64_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_f32_i32_q4_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q4_0,   quantize_q4_0>;
template [[host_name("kernel_set_rows_f32_i64_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_f32_i32_q4_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q4_1,   quantize_q4_1>;
template [[host_name("kernel_set_rows_f32_i64_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_f32_i32_q5_0")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q5_0,   quantize_q5_0>;
template [[host_name("kernel_set_rows_f32_i64_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_f32_i32_q5_1")]]   kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_q5_1,   quantize_q5_1>;
template [[host_name("kernel_set_rows_f32_i64_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<float, int64_t, block_iq4_nl, quantize_iq4_nl>;
template [[host_name("kernel_set_rows_f32_i32_iq4_nl")]] kernel set_rows_q32_t kernel_set_rows_q32<float, int32_t, block_iq4_nl, quantize_iq4_nl>;


typedef decltype(kernel_set_rows_q<float, int64_t, QK_K, block_tq2_0, quantize_tq2_0>) set_rows_qK_t;

template [[host_name("kernel_set_rows_f32_i64_tq2_0")]]  kernel set_rows_qK_t kernel_set_rows_q<float, int64_t, QK_K, block_tq2_0, quantize_tq2_0>;
template [[host_name("kernel_set_rows_f32_i32_tq2_0")]]  kernel set_rows_qK_t kernel_set_rows_q<float, int32_t, QK_K, block_tq2_0, quantize_tq2_0>;

// TurboQuant set_rows instantiations (128-element groups: 4x32-element blocks for turbo3, dedicated kernels for turbo2/4)
typedef decltype(kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>) set_rows_turbo3_t;
template [[host_name("kernel_set_rows_f32_i64_turbo3")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;
template [[host_name("kernel_set_rows_f32_i32_turbo3")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int32_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;

typedef decltype(kernel_set_rows_turbo2<int64_t>) set_rows_turbo2_t;
template [[host_name("kernel_set_rows_f32_i64_turbo2")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo2")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int32_t>;

typedef decltype(kernel_set_rows_turbo4<int64_t>) set_rows_turbo4_t;
template [[host_name("kernel_set_rows_f32_i64_turbo4")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int64_t>;
template [[host_name("kernel_set_rows_f32_i32_turbo4")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int32_t>;



// Shared pre-rotation kernels (used by both TQ3 and TQ4 for mul_mm path)
kernel void kernel_tq3_rotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = ((int64_t)tpig / 32) * 32;
    if (base >= n) return;

    float val = x[base + tiisg] * tq3_signs[tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        float other = simd_shuffle_xor(val, step);
        val = (tiisg & step) ? (other - val) : (other + val);
    }
    x[base + tiisg] = val * tq3_inv_sqrt32;
}

kernel void kernel_tq3_unrotate_act(
        device float * x [[buffer(0)]],
        constant int64_t & n [[buffer(1)]],
        uint tpig [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const int64_t base = ((int64_t)tpig / 32) * 32;
    if (base >= n) return;

    float val = x[base + tiisg];
    for (ushort step = 1; step < 32; step <<= 1) {
        float other = simd_shuffle_xor(val, step);
        val = (tiisg & step) ? (other - val) : (other + val);
    }
    x[base + tiisg] = val * tq3_inv_sqrt32 * tq3_signs[tiisg];
}

// ===== TurboQuant4 bulk dequant to fp16 (for prefill FA) =====
// Dequants turbo4 blocks → half buffer. Dispatch before f16 FA during prefill.
// Each thread processes one 128-element block.
kernel void kernel_turbo4_dequant_f16(
        device const block_turbo4_0 * src [[buffer(0)]],
        device       half           * dst [[buffer(1)]],
        constant     uint           & n_blocks [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint blk_idx = tgpig * ntg + tiitg;
    if (blk_idx >= n_blocks) return;

    device const block_turbo4_0 & blk = src[blk_idx];
    device half * out = dst + blk_idx * QK_TURBO4;
    const half norm_h = blk.norm;

    // 4-bit nibble unpack → centroid → scale by norm → write fp16
    for (int j = 0; j < QK_TURBO4; j += 2) {
        const uint8_t qb = blk.qs[j / 2];
        out[j    ] = turbo_centroids_4bit_h[(qb     ) & 0xF] * norm_h;
        out[j + 1] = turbo_centroids_4bit_h[(qb >> 4) & 0xF] * norm_h;
    }
}

// ===== TurboQuant Walsh-Hadamard Transform kernel =====
// O(d log d) rotation for 128-element groups. Replaces dense 128x128 matmul.
// Each thread processes one 128-element group using half4 vectorized butterfly.
// Uses the same WHT signs already defined (turbo_wht_signs1/2, turbo_wht_signs1_h4/2_h4).

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    // Each thread handles one 128-element group
    const int64_t group_idx = tgpig * ntg + tiitg;
    const int64_t n_groups = args.n_elements / 128;
    if (group_idx >= n_groups) return;

    const device float * in = src + group_idx * 128;
    device float * out = dst + group_idx * 128;

    // Load into half4 vectors for fast butterfly
    half4 v[32];
    const bool is_inverse = (args.direction == 1);

    // Apply first signs (s1 for fwd, s2 for inv)
    for (int i = 0; i < 32; i++) {
        float4 f = float4(in[i*4], in[i*4+1], in[i*4+2], in[i*4+3]);
        half4 s = is_inverse ? turbo_wht_signs2_h4[i] : turbo_wht_signs1_h4[i];
        v[i] = half4(f) * s;
    }

    // WHT butterfly (7 stages, vectorized half4)
    // h=1: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // h=2: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // h=4..64: between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i], b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }

    // Apply second signs + normalize, write output as fp32
    const half4 inv_sqrt = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        half4 s = is_inverse ? turbo_wht_signs1_h4[i] : turbo_wht_signs2_h4[i];
        float4 f = float4(v[i] * inv_sqrt * s);
        out[i*4]   = f.x;
        out[i*4+1] = f.y;
        out[i*4+2] = f.z;
        out[i*4+3] = f.w;
    }
}

