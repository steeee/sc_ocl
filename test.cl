
/* per element multiplication */

__kernel void mat_multiply( __global unsigned short* C, __global unsigned short* A, __global unsigned short* B, 
	   __local unsigned short* As, __local unsigned short* Bs, int uiWA, int uiWB, int trueLocalSize1)
{ 
    int x =  get_global_id(0);
    int y =  get_global_id(1);

    if (x< uiWA && y < uiWB)
        C[x*uiWB+y]  = A[x*uiWB+y] * B[x*uiWB+y];

}


/* per element subtraction */

__kernel void mat_sub( __global unsigned short* C,
                       __global unsigned short* A,
                       __global unsigned short* B,
                       int uiWA,
                       int uiWB)
                       
{ 

    int x =  get_global_id(0);
    int y =  get_global_id(1);

    if (x< uiWA && y < uiWB)
        C[x*uiWB+y]  = A[x*uiWB+y] - B[x*uiWB+y];
}


__kernel void hellocl(__global unsigned short *buffer)
{
    size_t gidx = get_global_id(0);
    size_t gidy = get_global_id(1);
    size_t lidx = get_local_id(0);
    size_t leng = get_global_size(0);

    //buffer[gidx + 4 * gidy] = (1 <<gidx)|(0x10<<gidy);
    buffer[gidx  +  leng * gidy] = gidx  +  leng * gidy;



}




#ifdef BUILD_BILATERAL


#ifdef BORDER_CONSTANT
#define ELEM(i,l_edge,r_edge,elem1,elem2) (i)<(l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(x, maxV) \
    { \
        x = max(min(x, maxV - 1), 0); \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(x, maxV) \
    { \
        if (x < 0) \
            x -= ((x - maxV + 1) / maxV) * maxV; \
        if (x >= maxV) \
            x %= maxV; \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#define EXTRAPOLATE_(x, maxV, delta) \
    { \
        if (maxV == 1) \
            x = 0; \
        else \
            do \
            { \
                if ( x < 0 ) \
                    x = -x - 1 + delta; \
                else \
                    x = maxV - 1 - (x - maxV) - delta; \
            } \
            while (x >= maxV || x < 0); \
    }
#ifdef BORDER_REFLECT
#define EXTRAPOLATE(x, maxV) EXTRAPOLATE_(x, maxV, 0)
#else
#define EXTRAPOLATE(x, maxV) EXTRAPOLATE_(x, maxV, 1)
#endif
#else
#error No extrapolation method
#endif




__kernel void
adaptiveBilateralFilter_C1_D0(
    __global const uchar * restrict src,
    __global uchar *dst,
    float alpha,
    int src_offset,
    int src_whole_rows,
    int src_whole_cols,
    int src_step,
    int dst_offset,
    int dst_rows,
    int dst_cols,
    int dst_step,
    __global const float * lut,
    int lut_step)
{
    int col = get_local_id(0);
    const int gX = get_group_id(0);
    const int gY = get_group_id(1);

    int src_x_off = (src_offset % src_step);
    int src_y_off = src_offset / src_step;
    int dst_x_off = (dst_offset % dst_step);
    int dst_y_off = dst_offset / dst_step;

    int startX = gX * (THREADS-ksX+1) - anX + src_x_off;
    int startY = (gY * (1+EXTRA)) - anY + src_y_off;

    int dst_startX = gX * (THREADS-ksX+1) + dst_x_off;
    int dst_startY = (gY * (1+EXTRA)) + dst_y_off;

    int posX = dst_startX - dst_x_off + col;
    int posY = (gY * (1+EXTRA))	;

    __local uchar data[ksY+EXTRA][THREADS];

    float tmp_sum[1+EXTRA];
    for(int tmpint = 0; tmpint < 1+EXTRA; tmpint++)
    {
        tmp_sum[tmpint] = (float)(0);
    }

#ifdef BORDER_CONSTANT
    bool con;
    uchar ss;
    for(int j = 0;	j < ksY+EXTRA; j++)
    {
        con = (startX+col >= 0 && startX+col < src_whole_cols && startY+j >= 0 && startY+j < src_whole_rows);

        int cur_col = clamp(startX + col, 0, src_whole_cols);
        if(con)
        {
            ss = src[(startY+j)*(src_step) + cur_col];
        }

        data[j][col] = con ? ss : 0;
    }
#else
    for(int j= 0; j < ksY+EXTRA; j++)
    {
        int selected_row = startY+j, selected_col = startX+col;
        EXTRAPOLATE(selected_row, src_whole_rows)
        EXTRAPOLATE(selected_col, src_whole_cols)

        data[j][col] = src[selected_row * (src_step) + selected_col];
    }
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    float var[1+EXTRA];

    float weight;
    float totalWeight = 0;

    int currValCenter;
    int currWRTCenter;

    int sumVal = 0;
    int sumValSqr = 0;

    if(col < (THREADS-(ksX-1)))
    {
        int currVal;

        int howManyAll = (2*anX+1)*(ksY);

        //find variance of all data
        int startLMj;
        int endLMj;

        // Top row: don't sum the very last element
        for(int extraCnt=0; extraCnt<=EXTRA; extraCnt++)
        {
#if CALCVAR
            startLMj = extraCnt;
            endLMj =  ksY+extraCnt-1;
            sumVal = 0;
            sumValSqr =0;
            for(int j = startLMj; j < endLMj; j++)
            {
                for(int i=-anX; i<=anX; i++)
                {
                    currVal	= (uint)(data[j][col+anX+i])	;

                    sumVal += currVal;
                    sumValSqr += mul24(currVal, currVal);
                }
            }
            var[extraCnt] =  clamp((float)( ( (sumValSqr * howManyAll)- mul24(sumVal , sumVal) ) ) /  ( (float)(howManyAll*howManyAll) ) , 0.1f, (float)(MAX_VAR_VAL) );
#else
            var[extraCnt] = (float)(MAX_VAR_VAL);
#endif
        }

        for(int extraCnt = 0; extraCnt <= EXTRA; extraCnt++)
        {

            // top row: include the very first element, even on first time
            startLMj = extraCnt;
            // go all the way, unless this is the last local mem chunk,
            // then stay within limits - 1
            endLMj =  extraCnt + ksY;

            // Top row: don't sum the very last element
            currValCenter = (int)( data[ (startLMj + endLMj)/2][col+anX] );

            for(int j = startLMj, lut_j = 0; j < endLMj; j++, lut_j++)
            {
                for(int i=-anX; i<=anX; i++)
                {
#if FIXED_WEIGHT
                    weight = 1.0f;
#else
                    currVal	= (int)(data[j][col+anX+i])	;
                    currWRTCenter = currVal-currValCenter;

#if ABF_GAUSSIAN
                    weight = exp( -0.5f * (float)mul24(currWRTCenter,currWRTCenter)/var[extraCnt]) * lut[lut_j*lut_step+anX+i] ;
#else
                    weight = var[extraCnt] / (var[extraCnt] + (float)mul24(currWRTCenter,currWRTCenter)) * lut[lut_j*lut_step+anX+i] ;
#endif
#endif
                    tmp_sum[extraCnt] += (float)(data[j][col+anX+i] * weight);
                    totalWeight += weight;
                }
            }

            if(posX >= 0 && posX < dst_cols && (posY+extraCnt) >= 0 && (posY+extraCnt) < dst_rows)
            {
                dst[(dst_startY+extraCnt) * (dst_step)+ dst_startX + col] = convert_uchar_rtz(tmp_sum[extraCnt]/totalWeight+0.5f);
            }

            totalWeight = 0;
        }
    }
}

#define POW2(a) ((a) * (a))
kernel void bilateral_filter(global float4 *in,
                             global float4 *out,
                             const  float radius,
                             const  float preserve)
{
    int gidx       = get_global_id(0);
    int gidy       = get_global_id(1);
    int n_radius   = ceil(radius);
    int dst_width  = get_global_size(0);
    int src_width  = dst_width + n_radius * 2;

    int u, v, i, j;
    float4 center_pix =
        in[(gidy + n_radius) * src_width + gidx + n_radius];
    float4 accumulated = 0.0f;
    float4 tempf       = 0.0f;
    float  count       = 0.0f;
    float  diff_map, gaussian_weight, weight;

    for (v = -n_radius;v <= n_radius; ++v)
    {
        for (u = -n_radius;u <= n_radius; ++u)
        {
            i = gidx + n_radius + u;
            j = gidy + n_radius + v;

            int gid1d = i + j * src_width;
            tempf = in[gid1d];

            diff_map = exp (
                - (   POW2(center_pix.x - tempf.x)
                    + POW2(center_pix.y - tempf.y)
                    + POW2(center_pix.z - tempf.z))
                * preserve);

            gaussian_weight =
                exp( - 0.5f * (POW2(u) + POW2(v)) / radius);

            weight = diff_map * gaussian_weight;

            accumulated += tempf * weight;
            count += weight;
        }
    }
    out[gidx + gidy * dst_width] = accumulated / count;
}
#endif //build_bilateral
