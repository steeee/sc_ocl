#include <opencv2/opencv.hpp>
#include "cstdio"
#include "CL/cl.h"

#define MAX_SOURCE_SIZE (0x100000)
#undef DEBUG_VERBOSE


#define CL_CHECK(_expr)                                                         \
    do {                                                                        \
        _expr;                                                                  \
        if (sc_ocl::ret == CL_SUCCESS)                                        \
        break;                                                                  \
        fprintf(stderr, "OpenCL Error: '%s' returned %d! line %d\n",            \
                                      #_expr, (int)sc_ocl::ret, __LINE__);    \
        abort();                                                                \
    } while (0)

enum ocl_kernel_func{ oclSUB = 0,
                      oclMUL,
                      oclADD,
                      oclDIV,
                      oclBILAT,
                      oclADBILAT,
                      oclMORPH
                    };


//! type of morphological operation
enum { MORPH_ERODE=CV_MOP_ERODE, MORPH_DILATE=CV_MOP_DILATE,
       MORPH_OPEN=CV_MOP_OPEN, MORPH_CLOSE=CV_MOP_CLOSE,
       MORPH_GRADIENT=CV_MOP_GRADIENT, MORPH_TOPHAT=CV_MOP_TOPHAT,
       MORPH_BLACKHAT=CV_MOP_BLACKHAT };

struct resolution{int width; int height; int channel;};

class sc_ocl {
public:
    sc_ocl();
    ~sc_ocl(){printf("***  destructor exiting   ***\n\n");};

    int readImage(const char*);

    /* copy data to write buffer */
    int uploadToDevice(const cv::Mat);
    int uploadConfidenceToDevice(const cv::Mat);

    /* build cl program from source */
    /* src path and build options */
    int buildProgram(const char*,  const char*);

    /* create kernel we want to execute */
    int createKernelFun(const ocl_kernel_func);

    /* Setup kernel arguments */
    int setupKernelArg(const int , const int , const void* );

    int readCalibShort(const char *);
    int readCalibLong(const char *);

    int finish();

    int adaptiveBilateralFilter( unsigned short *src,
                                 unsigned short *dst,
                                 int ksize_x,
                                 int ksize_y,
                                 double sigmaSpace,
                                 double maxSigmaColor = 20.0,
                                 int anchor_x = -1,
                                 int anchor_y = -1);

    int cl_bilateral_filter (
            cl_mem in_tex,
            cl_mem out_tex,
            size_t global_worksize,
            float  radius,
            float  preserve);

    inline void normalizeAnchor(int &anchor, int ksize)
    {
        if (anchor < 0)
            anchor = ksize >> 1;

        assert(0 <= anchor && anchor < ksize);
    }

    inline size_t roundUp(size_t sz, size_t n)
    {
        // we don't assume that n is a power of 2 (see alignSize)
        // equal to divUp(sz, n) * n
        size_t t = sz + n - 1;
        size_t rem = t % n;
        size_t result = t - rem;
        return result;
    }

    /* remove const due to the request of clCreateBuffer() */
    void GPUErode( const cl_mem *src,
                   cl_mem *dst,
                   const cl_mem *mat_kernel,
                   int ksize_x, int ksize_y,
                   const int anchor_x, const int anchor_y,
                   bool rectKernel);

    void GPUDilate( const cl_mem *src,
                    cl_mem *dst,
                    const cl_mem *mat_kernel,
                    int ksize_x, int ksize_y,
                    const int anchor_x, const int anchor_y,
                    bool rectKernel );

    int morphologyEx( const cv::Mat &src,
                      cv::Mat &dst,
                      int operation,
                      const cv::Mat &element,
                      cv::Point anchor = cv::Point(-1, -1),
                      int morph_iteration = 1 );

    void erode( const cv::Mat &src,
                cv::Mat &dst,
                const cv::Mat &kernel,
                cv::Point anchor,
                int iterations,
                int borderType,
                const cv::Scalar &borderValue );

    void dilate( const cv::Mat &src,
                 cv::Mat &dst,
                 const cv::Mat &kernel,
                 cv::Point anchor,
                 int iterations,
                 int borderType,
                 const cv::Scalar &borderValue );

    void morphOp( int op,
                  const cv::Mat &src,
                  cv::Mat &dst,
                  const cv::Mat &_kernel,
                  int ksize_x,
                  int ksize_y,
                  cv::Point anchor,
                  int iterations,
                  int borderType,
                  const cv::Scalar &borderValue );

    int doMorphOp( int op,
                    const cv::Mat &src,
                    cv::Mat &dst,
                    int type,
                    const cv::Mat &_kernel,
                    const cv::Size &ksize,
                    cv::Point anchor,
                    int iterations );

    /* test function */
    int test();
    int test_bilateral();
    int test_morph();

    /* global return status */
    cl_int ret;
private:

    /* image parameters */
    int i_width;
    int i_height;
    int i_channel;
    int i_scalar;
    int i_pixSize;
    int i_type;
    int i_step;
    void *data;

    /* ocl info */
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue depth_q = NULL;

    /* phase and confidence data */
    cl_mem clMatPhase = NULL;
    cl_mem clMatConfidence = NULL;
    cl_mem clMatThita = NULL;

    cl_mem clTestOutput = NULL;
    cl_mem clTestInput1 = NULL;
    cl_mem clTestInput2 = NULL;

    cl_program program = NULL;
    cl_program depth_program = NULL;
    cl_kernel kernel = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
};

sc_ocl::sc_ocl()
{
    /* image resolution */
    i_width = 160;
    i_height = 120;

    /* Get platform/device information */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    /* Create OpenCL Context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    depth_q = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Create buffer object */
    clMatPhase = clCreateBuffer(context, CL_MEM_READ_WRITE, i_width * i_height* sizeof(unsigned short), NULL, &ret);
    clMatConfidence = clCreateBuffer(context, CL_MEM_READ_WRITE, i_width * i_height * sizeof(unsigned short), NULL, &ret);
    clMatThita = clCreateBuffer(context, CL_MEM_READ_WRITE, i_width * i_height * sizeof(unsigned short), NULL, &ret);

}


int sc_ocl::uploadToDevice(const cv::Mat inMat)
{

    /* Copy input data to memory buffer */
    ret = clEnqueueWriteBuffer(depth_q, clMatPhase, CL_TRUE,
                                 0, 160 * 120 * sizeof(unsigned short), inMat.data, 0, NULL, NULL);
    return ret;
}

int sc_ocl::uploadConfidenceToDevice(const cv::Mat inMat)
{

    /* Copy input data to memory buffer */
    ret = clEnqueueWriteBuffer(depth_q, clMatConfidence, CL_TRUE,
                                 0, 160 * 120 * sizeof(unsigned short), inMat.data, 0, NULL, NULL);
    return ret;
}

int sc_ocl::buildProgram(const char *src, const char* build_option)
{

    if (kernel != NULL)
        clReleaseKernel(kernel);

    if (program != NULL)
        clReleaseProgram(program);

    /* Load the kernel source code into the array source_str */
    FILE *fp;
    char *source_str = NULL;
    size_t source_size;

    //fp = fopen("sc_kernel.cl", "r");
    //fp = fopen("matrixMul.cl", "r");
    fp = fopen(src, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    if(!source_str)
        fprintf(stderr, "Failed to alloc kernel source.\n");
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create a program from the kernel source */
    program = clCreateProgramWithSource(context, 1,
            (const char **)&source_str, (const size_t *)&source_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, build_option, NULL, NULL);

    if (ret != CL_SUCCESS) {
        char buffer[10240];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, NULL);
        fprintf(stderr, "CL Compilation failed:\n%s", buffer);
        abort();
    }

    free(source_str);

    return ret;
}

int sc_ocl::createKernelFun(const ocl_kernel_func func)
{

    /* Create the OpenCL kernel */
    switch (func) {
    case oclSUB:
        kernel = clCreateKernel(program, "mat_sub", &ret);
        break;
    case oclADD:
        kernel = clCreateKernel(program, "mat_add", &ret);
        break;
    case oclMUL:
        kernel = clCreateKernel(program, "mat_multiply", &ret);
        break;
    case oclBILAT:
        kernel = clCreateKernel(program, "bilateral_filter", &ret);
        break;
    case oclADBILAT:
        kernel = clCreateKernel(program, "adaptiveBilateralFilter_C1_D0", &ret);
        break;
    case oclMORPH:
        kernel = clCreateKernel(program, "morph_C1_D0", &ret);
        break;
    default:
        printf("no known function\n");
    }
    return ret;
}

/* provide kernel arguments to kernel, caller should manually provide the arguments in sequence */
int sc_ocl::setupKernelArg(const int index, const int size, const void* data)
{

    /* Set the arguments of the kernel */
    ret = clSetKernelArg(kernel, index, size, data);
    return ret;
}

int sc_ocl::adaptiveBilateralFilter( unsigned short *src, unsigned short *dst, int ksize_x, int ksize_y, double sigmaSpace, double maxSigmaColor, int anchor_x, int anchor_y)
{

    /* ksize must be odd */
    assert((ksize_x& 1) && (ksize_y & 1));
    /* source must be b bit RGB image */
    assert(i_type == CV_8UC1 || i_type == CV_8UC3);

    int ABF_GAUSSIAN_ocl = 1;

    char btype[30];
    sprintf(btype, "BORDER_REFLECT_101");
    int w = ksize_x / 2;
    int h = ksize_y / 2;
    int idx = 0;
    double sigma2 = sigmaSpace * sigmaSpace;

    float lut[ ksize_x * ksize_y ];

    int src_size = i_width * i_height * i_pixSize;
    cl_mem clTestOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, src_size, NULL, &ret);
    cl_mem clTestInput1 = clCreateBuffer(context, CL_MEM_READ_WRITE, src_size, src, &ret);
    cl_mem clDlut = clCreateBuffer(context, CL_MEM_READ_WRITE, ksize_x * ksize_y * sizeof(float), lut, &ret);

    for(int y=-h; y<=h; y++) {
        for(int x=-w; x<=w; x++) {
            lut[idx++] = expf( (float)(-0.5 * (x * x + y * y)/sigma2));
#ifdef DEBUG_VERBOSE
            fprintf(stderr, "%.4f ", lut[idx-1]);
#endif
        }
#ifdef DEBUG_VERBOSE
        fprintf(stderr, "\n");
#endif
    }
    clEnqueueWriteBuffer(depth_q, clTestInput1, CL_TRUE, 0, src_size, src, 0,NULL, NULL);
    clEnqueueWriteBuffer(depth_q, clTestOutput, CL_TRUE, 0, src_size, dst, 0,NULL, NULL);
    clEnqueueWriteBuffer(depth_q, clDlut, CL_TRUE, 0, ksize_x * ksize_y * sizeof(float), lut, 0,NULL, NULL);

    normalizeAnchor(anchor_x, ksize_x);
    normalizeAnchor(anchor_y, ksize_y);

    //the following constants may be adjusted for performance concerns
    const static size_t blockSizeX = 64, blockSizeY = 1, EXTRA = ksize_y - 1;

    //Normalize the result by default
    const float alpha = ksize_x * ksize_y;

    const size_t gSize = blockSizeX - ksize_x / 2 * 2;
    const size_t globalSizeX = (i_width) % gSize == 0 ?
                                i_width / gSize * blockSizeX :
                                (i_width / gSize + 1) * blockSizeX;

    const size_t rows_per_thread = 1 + EXTRA;
    const size_t globalSizeY = ((i_height + rows_per_thread - 1) / rows_per_thread) % blockSizeY == 0 ?
                               ((i_height + rows_per_thread - 1) / rows_per_thread) :
                               (((i_height + rows_per_thread - 1) / rows_per_thread) / blockSizeY + 1) * blockSizeY;

    size_t globalThreads[3] = { globalSizeX, globalSizeY, 1};
    size_t localThreads[3]  = { blockSizeX, blockSizeY, 1};

    char build_options[250];
    sprintf(build_options,
        "-D BUILD_BILATERAL -D VAR_PER_CHANNEL=1 -D CALCVAR=1 -D FIXED_WEIGHT=0 -D EXTRA=%d -D MAX_VAR_VAL=%f -D ABF_GAUSSIAN=%d"
        " -D THREADS=%d -D anX=%d -D anY=%d -D ksX=%d -D ksY=%d -D %s",
        static_cast<int>(EXTRA), static_cast<float>(maxSigmaColor*maxSigmaColor), static_cast<int>(ABF_GAUSSIAN_ocl),
        static_cast<int>(blockSizeX), anchor_x, anchor_y, ksize_x, ksize_y, btype);

#ifdef DEBUG_VERBOSE
    fprintf(stderr, "build option is %s", build_options);
#endif
    CL_CHECK(buildProgram("test.cl", build_options));
    CL_CHECK(createKernelFun(oclADBILAT));

    int src_offset = 0;
    int step = i_step;

#ifdef DEBUG_VERBOSE
    fprintf(stderr, "gsize  is %ld\n", gSize);
    fprintf(stderr, "global x is %ld\n", globalSizeX);
    fprintf(stderr, "global y is %ld\n", globalSizeY);
    fprintf(stderr, "local x is %ld\n", blockSizeX);
    fprintf(stderr, "local y is %ld\n", blockSizeY);

    fprintf(stderr, "alpha is %f\n", alpha);
    fprintf(stderr, "src offsetis %d\n", src_offset);
    fprintf(stderr, "src wholerowsis %d\n", i_height);
    fprintf(stderr, "src wholecolsis %d\n", i_width);
    fprintf(stderr, "src stepis %d\n", step);
    fprintf(stderr, "dst.offset is %d\n", src_offset);
    fprintf(stderr, "dst.row is %d\n", i_height);
    fprintf(stderr, "dst.cols is %d\n", i_width);
    fprintf(stderr, "dst.steps is %d\n", step);
#endif

    CL_CHECK(setupKernelArg(0,  sizeof(cl_mem),   (void*)&clTestInput1));
    CL_CHECK(setupKernelArg(1,  sizeof(cl_mem),   (void*)&clTestOutput));
    CL_CHECK(setupKernelArg(2,  sizeof(cl_float), (void*)&alpha));
    CL_CHECK(setupKernelArg(3,  sizeof(cl_int),   (void*)&src_offset));
    CL_CHECK(setupKernelArg(4,  sizeof(cl_int),   (void*)&i_height));
    CL_CHECK(setupKernelArg(5,  sizeof(cl_int),   (void*)&i_width));
    CL_CHECK(setupKernelArg(6,  sizeof(cl_int),   (void*)&step));
    CL_CHECK(setupKernelArg(7,  sizeof(cl_int),   (void*)&src_offset));
    CL_CHECK(setupKernelArg(8,  sizeof(cl_int),   (void*)&i_height));
    CL_CHECK(setupKernelArg(9,  sizeof(cl_int),   (void*)&i_width));
    CL_CHECK(setupKernelArg(10, sizeof(cl_int),   (void*)&step));
    int lut_step = 4;
    /* int lut_step = 8; //seems both 4 and 8 are OK */
    CL_CHECK(setupKernelArg(11, sizeof(cl_mem),   (void*)&clDlut));
    CL_CHECK(setupKernelArg(12, sizeof(cl_int),   (void*)&lut_step));

    if ( localThreads != NULL)
    {
        globalThreads[0] = roundUp(globalThreads[0], localThreads[0]);
        globalThreads[1] = roundUp(globalThreads[1], localThreads[1]);
        globalThreads[2] = roundUp(globalThreads[2], localThreads[2]);
    }

    cl_event event = NULL;
    CL_CHECK(clEnqueueNDRangeKernel(depth_q, kernel, 3, NULL, globalThreads,
            localThreads, 0, NULL, &event));

    clWaitForEvents(1, &event);
    CL_CHECK(clFinish(depth_q));
    CL_CHECK(clEnqueueReadBuffer(depth_q,
            clTestOutput, CL_TRUE, 0, i_height * i_width * i_pixSize,
            dst, 0, NULL, NULL));
    clReleaseEvent(event);

   return 1;
}

int sc_ocl::cl_bilateral_filter (
                     cl_mem in_tex,
                     cl_mem out_tex,
                     size_t global_worksize,
                     float  radius,
                     float  preserve)
{
  size_t global_ws[2];

  createKernelFun(oclBILAT);
  global_ws[0] = i_width;
  global_ws[1] = i_height;

  setupKernelArg(0, sizeof(cl_mem),   (void*)&in_tex);
  setupKernelArg(1, sizeof(cl_mem),   (void*)&out_tex);
  setupKernelArg(2, sizeof(cl_float), (void*)&radius);
  setupKernelArg(3, sizeof(cl_float), (void*)&preserve);

  ret = clEnqueueNDRangeKernel(depth_q,
                               kernel, 2,
                               NULL, global_ws, NULL,
                               0, NULL, NULL);
  return 0;
}

/*
** We should be able to support any data types here.
** Extend this if necessary later.
** Note that the kernel need to be further refined.
*/
void sc_ocl::GPUErode( const cl_mem *src,
                         cl_mem *dst,
                         const cl_mem *mat_kernel,
                         int ksize_x, int ksize_y,
                         const int anchor_x, const int anchor_y,
                         bool rectKernel)
{
    //Normalize the result by default
    //float alpha = ksize.height * ksize.width;

    fprintf(stderr, "i_step is %d, i_pixSize is %d\n", i_step, i_pixSize);
    int srcStep = i_step / i_pixSize;
    int dstStep = srcStep;
    int dstOffset = 0;

    int srcOffset_x = 0;
    int srcOffset_y = 0;

#ifdef ANDROID
    size_t localThreads[3] = {16, 8, 1};
#else
    size_t localThreads[3] = {16, 16, 1};
#endif
    size_t globalThreads[3] = {(i_width + localThreads[0] - 1) / localThreads[0] *localThreads[0], (i_height + localThreads[1] - 1) / localThreads[1] *localThreads[1], 1};

    if (i_type == CV_8UC1)
    {
        globalThreads[0] = ((i_width + 3) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
        assert(localThreads[0]*localThreads[1] * 8 >=
                   ((localThreads[0] * 4 + ksize_x- 1) * (localThreads[1] + ksize_y - 1)));
    }
    else
    {
        fprintf(stderr, "*** format is NOT cv_8UC1, result maybe incorrect ***\n");
        assert(localThreads[0]*localThreads[1] * 2 >=
                  ((localThreads[0] + ksize_x - 1) * (localThreads[1] + ksize_y - 1)));
    }

    char s[64];
    switch (i_type)
    {
    case CV_8UC1:
        sprintf(s, "-D VAL=255");
        break;
    case CV_8UC3:
    case CV_8UC4:
        sprintf(s, "-D VAL=255 -D GENTYPE=uchar4");
        break;
    case CV_32FC1:
        sprintf(s, "-D VAL=FLT_MAX -D GENTYPE=float");
        break;
    case CV_32FC3:
    case CV_32FC4:
        sprintf(s, "-D VAL=FLT_MAX -D GENTYPE=float4");
        break;
    default:
        fprintf(stderr, "unsupported type");
        break;
    }

    char compile_option[128];
    sprintf(compile_option, "-D RADIUSX=%d -D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D ERODE %s %s",
        anchor_x, anchor_y, (int)localThreads[0], (int)localThreads[1],
        s, rectKernel?"-D RECTKERNEL":"");

#ifdef DEBUG_VERBOSE
    fprintf(stderr, "build option is %s\n", compile_option);
#endif
    CL_CHECK(buildProgram("filtering_morph.cl", compile_option));
    CL_CHECK(createKernelFun(oclMORPH));

    CL_CHECK(setupKernelArg(0,  sizeof(cl_mem), (void*)src));
    CL_CHECK(setupKernelArg(1,  sizeof(cl_mem), (void*)dst));
    CL_CHECK(setupKernelArg(2,  sizeof(cl_int), (void*)&srcOffset_x));
    CL_CHECK(setupKernelArg(3,  sizeof(cl_int), (void*)&srcOffset_y));
    CL_CHECK(setupKernelArg(4,  sizeof(cl_int), (void*)&i_width));
    CL_CHECK(setupKernelArg(5,  sizeof(cl_int), (void*)&i_height));
    CL_CHECK(setupKernelArg(6,  sizeof(cl_int), (void*)&srcStep));
    CL_CHECK(setupKernelArg(7,  sizeof(cl_int), (void*)&dstStep));
    CL_CHECK(setupKernelArg(8,  sizeof(cl_mem), (void*)mat_kernel));
    CL_CHECK(setupKernelArg(9,  sizeof(cl_int), (void*)&i_width));
    CL_CHECK(setupKernelArg(10, sizeof(cl_int), (void*)&i_height));
    CL_CHECK(setupKernelArg(11, sizeof(cl_int), (void*)&dstOffset));

    cl_event event = NULL;
    CL_CHECK(clEnqueueNDRangeKernel(depth_q, kernel, 3, NULL, globalThreads,
            localThreads, 0, NULL, &event));

    clWaitForEvents(1, &event);
    CL_CHECK(clFinish(depth_q));
    clReleaseEvent(event);
}


//! data type supported: CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4
void sc_ocl::GPUDilate( const cl_mem *src,
                          cl_mem *dst,
                          const cl_mem *mat_kernel,
                          int ksize_x, int ksize_y,
                          const int anchor_x, const int anchor_y,
                          bool rectKernel )
{
    int srcStep = i_step / i_pixSize;
    int dstStep = srcStep;
    int dstOffset = 0;

    int srcOffset_x = 0;
    int srcOffset_y = 0;

#ifdef ANDROID
    size_t localThreads[3] = {16, 10, 1};
#else
    size_t localThreads[3] = {16, 16, 1};
#endif
    size_t globalThreads[3] = {(i_width + localThreads[0] - 1) / localThreads[0] *localThreads[0],
                               (i_height + localThreads[1] - 1) / localThreads[1] *localThreads[1], 1};

    fprintf(stderr, "testpoint %d\n", __LINE__);
    if (i_type == CV_8UC1)
    {
        globalThreads[0] = ((i_width + 3) / 4 + localThreads[0] - 1) / localThreads[0] * localThreads[0];
        assert(localThreads[0]*localThreads[1] * 8 >= (localThreads[0] * 4 + ksize_x - 1) * (localThreads[1] + ksize_y - 1));
    }
    else
    {
        fprintf(stderr, "*** format is NOT cv_8UC1, result maybe incorrect ***\n");
        assert(localThreads[0]*localThreads[1] * 2 >= (localThreads[0] + ksize_x - 1) * (localThreads[1] + ksize_y - 1));
    }

    char s[64];

    switch (i_type)
    {
    case CV_8UC1:
        sprintf(s, "-D VAL=0");
        break;
    case CV_8UC3:
    case CV_8UC4:
        sprintf(s, "-D VAL=0 -D GENTYPE=uchar4");
        break;
    case CV_32FC1:
        sprintf(s, "-D VAL=-FLT_MAX -D GENTYPE=float");
        break;
    case CV_32FC3:
    case CV_32FC4:
        sprintf(s, "-D VAL=-FLT_MAX -D GENTYPE=float4");
        break;
    default:
        CV_Error(CV_StsUnsupportedFormat, "unsupported type");
    }

    char compile_option[128];
    sprintf(compile_option, "-D RADIUSX=%d -D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D DILATE %s %s",
        anchor_x, anchor_y, (int)localThreads[0], (int)localThreads[1],
        s, rectKernel?"-D RECTKERNEL":"");

#ifdef DEBUG_VERBOSE
    fprintf(stderr, "build option is %s", compile_option);
#endif

    CL_CHECK(buildProgram("filtering_morph.cl", compile_option));
    CL_CHECK(createKernelFun(oclMORPH));

    CL_CHECK(setupKernelArg(0,  sizeof(cl_mem), (void*)src));
    CL_CHECK(setupKernelArg(1,  sizeof(cl_mem), (void*)dst));
    CL_CHECK(setupKernelArg(2,  sizeof(cl_int), (void*)&srcOffset_x));
    CL_CHECK(setupKernelArg(3,  sizeof(cl_int), (void*)&srcOffset_y));
    CL_CHECK(setupKernelArg(4,  sizeof(cl_int), (void*)&i_width));
    CL_CHECK(setupKernelArg(5,  sizeof(cl_int), (void*)&i_height));
    CL_CHECK(setupKernelArg(6,  sizeof(cl_int), (void*)&srcStep));
    CL_CHECK(setupKernelArg(7,  sizeof(cl_int), (void*)&dstStep));
    CL_CHECK(setupKernelArg(8,  sizeof(cl_mem), (void*)mat_kernel));
    CL_CHECK(setupKernelArg(9,  sizeof(cl_int), (void*)&i_width));
    CL_CHECK(setupKernelArg(10, sizeof(cl_int), (void*)&i_height));
    CL_CHECK(setupKernelArg(11, sizeof(cl_int), (void*)&dstOffset));

    cl_event event = NULL;
    CL_CHECK(clEnqueueNDRangeKernel(depth_q, kernel, 3, NULL, globalThreads,
            localThreads, 0, NULL, &event));

    clWaitForEvents(1, &event);
    CL_CHECK(clFinish(depth_q));
    CL_CHECK(clEnqueueReadBuffer(depth_q,
            clTestOutput, CL_TRUE, 0, i_height * i_width * i_pixSize,
            dst, 0, NULL, NULL));
    clReleaseEvent(event);
}


int sc_ocl::doMorphOp( int op,
                         const cv::Mat &src,
                         cv::Mat &dst,
                         int type,
                         const cv::Mat &_kernel,
                         const cv::Size &ksize,
                         cv::Point anchor,
                         int iterations )
{
    assert(op == cv::MORPH_ERODE || op == cv::MORPH_DILATE);
    assert(type == CV_8UC1 || type == CV_8UC3 ||
           type == CV_8UC4 || type == CV_32FC1 ||
           type == CV_32FC3 || type == CV_32FC4);

    cv::Mat kernel8U;
    _kernel.convertTo(kernel8U, CV_8U);
    cv::Mat kernel = kernel8U.reshape(1, 1);

/* not sure if we have to keep this for loop, guess not */
    for(int i = 0; i < kernel.rows * kernel.cols; ++i) {
        if(kernel.at<uchar>(i) != 1) {
        }
    }

    unsigned short *input1 = (unsigned short*)src.data;
    unsigned short *lut = (unsigned short*)malloc(_kernel.elemSize() * _kernel.rows * _kernel.cols);
    memcpy(lut, _kernel.data, _kernel.elemSize() * _kernel.rows * _kernel.cols);

    i_width = src.cols;
    i_height = src.rows;
    i_pixSize = src.elemSize();
    i_type = src.type();
    i_step = src.step;
    dst = src.clone();

    int buf_size = i_width * i_height * i_pixSize;
    int kernel_size = _kernel.rows * _kernel.cols * _kernel.elemSize();

    cl_mem clTestOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, NULL, &ret);
    cl_mem clTestInput1 = clCreateBuffer(context, CL_MEM_READ_WRITE, buf_size, input1, &ret);
    cl_mem clDlut = clCreateBuffer(context, CL_MEM_READ_WRITE, kernel_size, lut, &ret);

    clEnqueueWriteBuffer(depth_q, clTestInput1, CL_TRUE, 0, buf_size, input1, 0,NULL, NULL);
    clEnqueueWriteBuffer(depth_q, clTestOutput, CL_TRUE, 0, buf_size, dst.data, 0,NULL, NULL);
    clEnqueueWriteBuffer(depth_q, clDlut, CL_TRUE, 0, kernel_size, lut, 0,NULL, NULL);

    int rectKernel = false;

    /* we use the reshaped kernel */
    if (op == MORPH_ERODE) {

        GPUErode(&clTestInput1, &clTestOutput, &clDlut,
                ksize.width, ksize.height, anchor.x, anchor.y, rectKernel);

        CL_CHECK(clEnqueueReadBuffer(depth_q,
                    clTestOutput, CL_TRUE, 0, i_height * i_width * i_pixSize,
                    dst.data, 0, NULL, NULL));

    } else { /* Dilate */

        GPUDilate(&clTestInput1, &clTestOutput, &clDlut,
                         ksize.width, ksize.height, anchor.x, anchor.y, rectKernel);

        CL_CHECK(clEnqueueReadBuffer(depth_q,
                    clTestOutput, CL_TRUE, 0, i_height * i_width * i_pixSize,
                    dst.data, 0, NULL, NULL));
    }

    free(lut);
    return 1;
}

void sc_ocl::morphOp( int op,
                        const cv::Mat &src,
                        cv::Mat &dst,
                        const cv::Mat &_kernel,
                        int ksize_x, int ksize_y,
                        cv::Point anchor,
                        int iterations, int borderType,
                        const cv::Scalar &borderValue )
{
    if ((borderType != cv::BORDER_CONSTANT) ||
             (borderValue != cv::morphologyDefaultBorderValue()))
    {
        fprintf(stderr, "unsupported border type\n");
        exit(1);
    }

    cv::Mat kernel;
    cv::Size ksize = _kernel.data ? _kernel.size() : cv::Size(3, 3);

    normalizeAnchor(anchor.x, ksize_x);
    normalizeAnchor(anchor.y, ksize_y);
    kernel = _kernel;

    doMorphOp(op, src, dst, src.type(), kernel, ksize, anchor, iterations);
}

void sc_ocl::erode( const cv::Mat &src,
                      cv::Mat &dst,
                      const cv::Mat &kernel,
                      cv::Point anchor,
                      int iterations,
                      int borderType,
                      const cv::Scalar &borderValue)
{
    bool allZero = true;

    for (int i = 0; i < kernel.rows * kernel.cols; ++i)
        if (kernel.data[i] != 0)
            allZero = false;

    if (allZero)
        kernel.data[0] = 1;

    morphOp(cv::MORPH_ERODE, src, dst, kernel,
            kernel.cols, kernel.rows, anchor, iterations, borderType, borderValue);
}

void sc_ocl::dilate( const cv::Mat &src,
                       cv::Mat &dst,
                       const cv::Mat &kernel, cv::Point anchor, int iterations,
                     int borderType, const cv::Scalar &borderValue)
{
    morphOp(MORPH_DILATE, src, dst, kernel, kernel.cols, kernel.rows,
            anchor, iterations, borderType, borderValue);
}

int sc_ocl::morphologyEx( const cv::Mat &src,
                            cv::Mat &dst,
                            int operation,
                            const cv::Mat &kernel,
                            cv::Point anchor,
                            int iterations)
{
    int borderType = cv::BORDER_CONSTANT;
    const cv::Scalar borderValue = cv::morphologyDefaultBorderValue();
    cv::Mat temp;

    switch (operation)
    {
    case MORPH_ERODE:
        erode(src, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case MORPH_DILATE:
        dilate(src, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case MORPH_OPEN:
        erode(src, temp, kernel, anchor, iterations, borderType, borderValue);
        dilate(temp, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case CV_MOP_CLOSE:
        dilate(src, temp, kernel, anchor, iterations, borderType, borderValue);
        erode(temp, dst, kernel, anchor, iterations, borderType, borderValue);
        break;
    case CV_MOP_GRADIENT:
        erode(src, temp, kernel, anchor, iterations, borderType, borderValue);
        dilate(src, dst, kernel, anchor, iterations, borderType, borderValue);
        subtract(dst, temp, dst);
        break;
    case CV_MOP_TOPHAT:
        erode(src, dst, kernel, anchor, iterations, borderType, borderValue);
        dilate(dst, temp, kernel, anchor, iterations, borderType, borderValue);
        subtract(src, temp, dst);
        break;
    case CV_MOP_BLACKHAT:
        dilate(src, dst, kernel, anchor, iterations, borderType, borderValue);
        erode(dst, temp, kernel, anchor, iterations, borderType, borderValue);
        subtract(temp, src, dst);
        break;
    default:
        fprintf(stderr, "unknown morphological operation");
        break;
    }
    return 1;
}

/* ********** test functions implementation ********** */
int sc_ocl::test_morph()
{
  //! shape of the structuring element
  /* enum { MORPH_RECT=0, MORPH_CROSS=1, MORPH_ELLIPSE=2 }; */
  int morph_elem = 2;
  int morph_size = 2;
  int morph_iteration = 1;

  //cv::Mat src = cv::imread("LenaRGB.bmp", 0);
  //cv::Mat src = cv::imread("len_full.jpg", 0);
  cv::Mat src = cv::imread("l_unpub.jpg", 0);
  cv::Mat src_resize, dst_erode, dst_dilate, dst_grade;
  if (! src.data) {
      fprintf(stderr, "no image found!!\n");
      exit(1);
  }

  cv::Mat element = cv::getStructuringElement( morph_elem, cv::Size( 3*morph_size + 1, 3*morph_size+1 ), cv::Point( morph_size, morph_size ) );

  morphologyEx( src, dst_erode, MORPH_ERODE, element, cv::Point(-1, -1), morph_iteration );
  morphologyEx( src, dst_dilate, MORPH_DILATE, element, cv::Point(-1, -1), morph_iteration );
  morphologyEx( src, dst_grade, MORPH_GRADIENT, element, cv::Point(-1, -1), morph_iteration );

  imshow("src", src);
  imshow("erode", dst_erode);
  imshow("dilate", dst_dilate);
  imshow("gradient", dst_grade);
  cv::moveWindow("src", 200+i_width, 400);
  cv::moveWindow("erode", 300+i_width, 400);
  cv::moveWindow("dilate", 400+i_width, 400);
  cv::moveWindow("gradient", 500+i_width, 400);
  cvWaitKey(1000000);
  return 1;
}

int sc_ocl::test_bilateral()
{

    cv::Mat src = cv::imread("len_full.jpg", 0);
    cv::Mat src_resize, dst;
    if (! src.data) {
        fprintf(stderr, "no image found!!\n");
        exit(1);
    }
    int steps = src.step;
    fprintf(stderr, "image row is %d!!\n", src.rows);
    fprintf(stderr, "image col is %d!!\n", src.cols);
    fprintf(stderr, "image channels is %d!!\n", src.channels());
    fprintf(stderr, "image depth is %d!!\n", src.depth());
    fprintf(stderr, "image size is %d!!\n", src.size().height*src.size().width);
    fprintf(stderr, "image elemsize1 is %ld!!\n", src.elemSize1());
    fprintf(stderr, "image elemsize is %ld!!\n", src.elemSize());

    for (int i = 0; i < src.dims; i++)
        fprintf(stderr, "image step[%d] is %ld!!\n", i, src.step[i]);

    fprintf(stderr, "image step is %d!!\n", steps);
    fprintf(stderr, "image step1 is %ld!!\n", src.step1());
    //    imshow("original", src);

    /* create a new image from src.data */
    //  src.convertTo(src_resize, CV_8UC1, 1);
    //  cv::Mat result(src.rows, src.cols, src.type(), src.data);
    //  imshow("output retult", result);

    src.convertTo(src, CV_8UC1, 1);
    fprintf(stderr, "image channels after converto is %d!!\n", src.channels());
    fprintf(stderr, "image elemsize after convertto is %ld!!\n", src.elemSize());
    fprintf(stderr, "image elemsize1 after convertto is %ld!!\n", src.elemSize1());
    fprintf(stderr, "image total after convertto is %ld!!\n", src.total());
    fprintf(stderr, "image type after convertto is %d!!\n", src.type());
    fprintf(stderr, "CV_8UC1 is %d!!\n", CV_8UC1);
    fprintf(stderr, "CV_16UC1 is %d!!\n", CV_16UC1);
    fprintf(stderr, "CV_8UC3 is %d!!\n", CV_8UC3);

    unsigned short *input1 = (unsigned short*)src.data;
    unsigned short *output = (unsigned short*)malloc(src.elemSize() * src.rows * src.cols);

//    memcpy(output, input1, src.elemSize() * src.rows * src.cols);

    i_width = src.cols;
    i_height = src.rows;
    i_pixSize = src.elemSize();
    i_type = src.type();
    i_step = src.step;
    fprintf(stderr, "i_pixSize is %d!!\n", i_pixSize);
    fprintf(stderr, "i_width is %d!!\n", i_width);
    fprintf(stderr, "i_height is %d!!\n", i_height);
 
    adaptiveBilateralFilter( input1, output, 11, 11, 200, 200);
    //kernel = clCreateKernel(program, "hellocl", &ret);
    
    cv::Mat result(i_height, i_width, i_type, output);
    imshow("output retult", result);
    cv::moveWindow("output retult", 400+i_width, 400);
    cv::Mat iresult(i_height, i_width, i_type, input1);
    imshow("input retult", iresult);
    cv::moveWindow("input retult", 400, 400);

    cv::waitKey(1000000);
    free(output);
    return 1;
}



int sc_ocl::test()
{

    unsigned short *input1 = (unsigned short*)malloc(sizeof(unsigned short) * 64);
    unsigned short *input2 = (unsigned short*)malloc(sizeof(unsigned short) * 64);
    unsigned short *output = (unsigned short*)malloc(sizeof(unsigned short) * 64);

    memset(input1, 0, sizeof(unsigned short)*64);
    memset(input2, 0, sizeof(unsigned short)*64);
    memset(output, 0, sizeof(unsigned short)*64);
    int width = 8;
    int height = 8;

    fprintf(stderr, "output::\n");
    for (int i = 0; i < width; i++) {

        for (int j = 0; j < height; j++){
                fprintf(stderr, "%d ", output[i*width+j]);
        }
        fprintf(stderr, "\n");
    }

    /* initialize input data */
    for (int i = 0; i < 64; i++){
        input1[i] = i+5;
        input2[i] = i;
    }

    /* prepare device buffer */
    clTestOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(unsigned short), NULL, &ret);
    clTestInput1 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(unsigned short), input1, &ret);
    clTestInput2 = clCreateBuffer(context, CL_MEM_READ_WRITE, 64 * sizeof(unsigned short), input2, &ret);

    clEnqueueWriteBuffer(depth_q, clTestInput1, CL_TRUE, 0, 64 * sizeof(unsigned short), input1, 0,NULL, NULL);
    clEnqueueWriteBuffer(depth_q, clTestInput2, CL_TRUE, 0, 64 * sizeof(unsigned short), input2, 0,NULL, NULL);
    clEnqueueWriteBuffer(depth_q, clTestOutput, CL_TRUE, 0, 64 * sizeof(unsigned short), output, 0,NULL, NULL);


    CL_CHECK(buildProgram("test.cl", NULL));
    CL_CHECK(createKernelFun(oclSUB));
    //kernel = clCreateKernel(program, "hellocl", &ret);

    CL_CHECK(setupKernelArg(0, sizeof(cl_mem), (void*)&clTestOutput));
    CL_CHECK(setupKernelArg(1, sizeof(cl_mem), (void*)&clTestInput1));
    CL_CHECK(setupKernelArg(2, sizeof(cl_mem), (void*)&clTestInput2));
    CL_CHECK(setupKernelArg(3, sizeof(int), (void*)&width));
    CL_CHECK(setupKernelArg(4, sizeof(int), (void*)&height));

    //ret = clEnqueueTask(depth_q, kernel, 0, NULL, NULL);
    size_t global_work_size[] = {8, 8};
    size_t local_work_size[] = {2, 2};
    ret = clEnqueueNDRangeKernel(depth_q, kernel, 2, NULL, global_work_size,
                                     local_work_size, 0, NULL, NULL);
    ret = clFinish(depth_q);

    /* Retrieve result from device */
    clEnqueueReadBuffer(depth_q,
            clTestOutput, CL_TRUE, 0, 64 * sizeof(unsigned short),
            output, 0, NULL, NULL);


    /* display result */
    fprintf(stderr, "input1::\n");
    for (int i = 0; i < width; i++) {

        for (int j = 0; j < height; j++){
                fprintf(stderr, "%d ", input1[i*width+j]);
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "input2::\n");
    for (int i = 0; i < width; i++) {

        for (int j = 0; j < height; j++){
                fprintf(stderr, "%d ", input2[i*width+j]);
        }
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "result::\n");
    for (int i = 0; i < width; i++) {

        for (int j = 0; j < height; j++){
                fprintf(stderr, "%d ", output[i*width+j]);
        }
        fprintf(stderr, "\n");
    }
   
    free(input1);
    free(input2);
    free(output);

    return 1;
}

int sc_ocl::finish()
{
    clReleaseMemObject(clMatPhase);
    clReleaseMemObject(clMatConfidence);
    clReleaseMemObject(clMatThita);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(depth_q);
    clReleaseContext(context);
    return 1;
}

int main(){
        fprintf(stderr, "initializing ocl object\n");
        sc_ocl a;
        fprintf(stderr, "running test functoins\n");
        a.test();
        fprintf(stderr, "running bilateral\n");
        a.test_bilateral();
        fprintf(stderr, "running morph\n");
        a.test_morph();
        fprintf(stderr, "finishing\n");
        a.finish();
        fprintf(stderr, "finished!!\n");
        return 0;
}
