#include <iostream>
#include <cstring>
#include <cmath>
#include <stdio.h>

#define GAUSS_KERNEL_SIZE            25
#define GAUSS_KERNEL_SIZE_BYTES            GAUSS_KERNEL_SIZE * sizeof(unsigned char)
#define GAUSS_KERNEL_SUM            159

#define SOBEL_KERNEL_SIZE            9
#define SOBEL_KERNEL_SIZE_BYTES            SOBEL_KERNEL_SIZE * sizeof(char)

#define BLOCK_SIZE_1D    32

// via http://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file
unsigned char *readBMP(const char *filename, int *width, int *height) {
	FILE *f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	*width = *(int *) &info[18];
	*height = *(int *) &info[22];

	int pixels = (*width) * (*height);
	int size = 3 * pixels;
	unsigned char *data = new unsigned char[size]; // allocate 3 bytes per pixel
	unsigned char *grayscale = new unsigned char[pixels]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);

	unsigned char *c = grayscale;

	//endianity (BGR -> RGB)
	for (int i = 0; i < size; i += 3) {
		*(c++) = (unsigned char) ((0.0722 * data[i] + 0.7152 * data[i + 1] + 0.2126 * data[i + 2]));
	}

	return grayscale;
}

void writeBMP(const char *filename, unsigned char *data, int width, int height) {
	int pixels = width * height;
	int size = 3 * pixels;
	unsigned char *colors = new unsigned char[size]; // allocate 3 bytes per pixel

	//endianity (BGR -> RGB)
	for (int i = 0; i < size; i += 3) {
		colors[i] = colors[i + 1] = colors[i + 2] = *(data++);
	}

	FILE *f = fopen(filename, "wb");
	//bmp header
	unsigned char info[54] = {'B', 'M', '6', 0xF9, 0x15, 0, 0, 0, 0, 0, 0x36, 0, 0, 0, 0x28, 0, 0, 0, 0x20, 0x3, 0, 0,
							  0x58, 0x02, 0, 0, 0x01, 0, 0x18, 0, 0, 0, 0, 0, 0, 0xF9, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0,
							  0, 0, 0, 0, 0, 0, 0};

	// write image height and width to header
	*(int *) &(info[18]) = width;
	*(int *) &(info[22]) = height;

	fwrite(info, sizeof(unsigned char), 54, f); // read the 54-byte header
	fwrite(colors, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);
}

void convolution(
		unsigned char *in,
		unsigned char *out,
		const int width,
		const int height,
		const char *kernel,
		const int kernelSize,
		const int kernelSum
) {
	const int offset = (kernelSize - 1) / 2;

	for (int x = offset; x < width - offset - 1; ++x) {
		for (int y = offset; y < height - offset - 1; ++y) {
			int newTotal = 0;
			for (int i = 0; i < kernelSize; ++i) {
				for (int j = 0; j < kernelSize; ++j) {
					newTotal += in[(y + j - offset) * height + x + i - offset] * kernel[j * kernelSize + i];
				}
			}
			out[y * height + x] = (unsigned char) (newTotal / kernelSum);
		}
	}
}

void convolutionWide(
		unsigned char *in,
		int *out,
		const int width,
		const int height,
		const char *kernel,
		const int kernelSize
) {
	const int offset = (kernelSize - 1) / 2;

	for (int x = offset; x < width - offset - 1; ++x) {
		for (int y = offset; y < height - offset - 1; ++y) {
			int newTotal = 0;
			for (int i = 0; i < kernelSize; ++i) {
				for (int j = 0; j < kernelSize; ++j) {
					newTotal += in[(y + j - offset) * height + x + i - offset] * kernel[j * kernelSize + i];
				}
			}
			out[y * height + x] = newTotal;
		}
	}
}


//gauss, |kernel|=5, sum=159
const unsigned char kernel[GAUSS_KERNEL_SIZE]{2, 4, 5, 4, 2,
							   4, 9, 12, 9, 4,
							   5, 12, 15, 12, 5,
							   4, 9, 12, 9, 4,
							   2, 4, 5, 4, 2};


//sobel directional
const char Gx[SOBEL_KERNEL_SIZE] = {-1, 0, 1,
					-2, 0, 2,
					-1, 0, 1};

const char Gy[SOBEL_KERNEL_SIZE] = {1, 2, 1,
					0, 0, 0,
					-1, -2, -1};


//	cudaReadModeElementType means no conversion on access time (optinally normalized)
texture<unsigned char, 2, cudaReadModeElementType> devImageTextureChar;
texture<float, 2, cudaReadModeElementType> devImageTextureFloat;

// the array bound to the 2D textures above
cudaArray* devImageChar, *devImageFloat;

// kernels/masks
__device__ __constant__ char devGxMask[9];
__device__ __constant__ char devGyMask[9];
__device__ __constant__ unsigned char devGaussMask[25];

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

void initDev(const unsigned int width, const unsigned int height)
{
	// initialize texture memory on the device (convolution kernels)
	gpuErrchk(cudaMemcpyToSymbol(devGaussMask, kernel, GAUSS_KERNEL_SIZE_BYTES));
	gpuErrchk(cudaMemcpyToSymbol(devGxMask, Gx, SOBEL_KERNEL_SIZE_BYTES));
	gpuErrchk(cudaMemcpyToSymbol(devGyMask, Gy, SOBEL_KERNEL_SIZE_BYTES));

	// tohle zpusobi, ze cteni indexu za hranou da nejblizsi platny pixel (a[-5] = a[0] napr.)
	devImageTextureChar.addressMode[0] = cudaAddressModeClamp;
	devImageTextureChar.addressMode[1] = cudaAddressModeClamp;
	devImageTextureFloat.addressMode[0] = cudaAddressModeClamp;
	devImageTextureFloat.addressMode[1] = cudaAddressModeClamp;


	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
	cudaMallocArray(&devImageChar, &channelDesc, width, height);

	cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float>();
	cudaMallocArray(&devImageFloat, &channelDesc2, width, height);
}


__global__ void devGaussKernel(unsigned char *output, const unsigned int width, const unsigned int height)
{
	const unsigned int	row = blockIdx.y * blockDim.y + threadIdx.y,
						col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int acc = 0;

	if (row < height && col < width)
	{
#pragma unroll
		for (char i = -2; i <= 2; ++i)
		{
			const unsigned int matrixColumn = col + i;
#pragma unroll
			for (char j = -2; j <= 2; ++j)
			{
				acc += devGaussMask[(i + 2) * 5 + (j + 2)] * (int)tex2D(devImageTextureChar, matrixColumn, row + j);
			}
		}

		output[row * width + col] = (unsigned char) ((float)acc / GAUSS_KERNEL_SUM);
	}
}

__global__ void devGradients(float *devOutGradients, float *devOutDirection, const unsigned int width, const unsigned int height)
{
	const unsigned int gRow = blockIdx.y * blockDim.y + threadIdx.y,
			gCol = blockIdx.x * blockDim.x + threadIdx.x,
			lRow = threadIdx.y + 1,
			lCol = threadIdx.x + 1;

	__shared__ unsigned char buffer[(BLOCK_SIZE_1D + 2) * (BLOCK_SIZE_1D + 2)];
	buffer[lCol * (BLOCK_SIZE_1D + 2) + lRow] = tex2D(devImageTextureChar, gCol, gRow);

	if (lRow == 1 || lCol == 1 || lCol == BLOCK_SIZE_1D || lRow == BLOCK_SIZE_1D)
	{
		buffer[(lCol - 1) * (BLOCK_SIZE_1D + 2) + lRow - 1] = tex2D(devImageTextureChar, gCol - 1, gRow - 1);
		buffer[(lCol - 1) * (BLOCK_SIZE_1D + 2) + lRow + 1] = tex2D(devImageTextureChar, gCol - 1, gRow + 1);
		buffer[(lCol + 1) * (BLOCK_SIZE_1D + 2) + lRow - 1] = tex2D(devImageTextureChar, gCol + 1, gRow - 1);
		buffer[(lCol + 1) * (BLOCK_SIZE_1D + 2) + lRow + 1] = tex2D(devImageTextureChar, gCol + 1, gRow + 1);
	}

	__syncthreads();

	int accX = 0, accY = 0;

	if (gRow < height && gCol < width)
	{
#pragma unroll
		for (int i = -1; i <= 1; ++i)
		{
			const unsigned int matrixColumn = (lCol + i) * (BLOCK_SIZE_1D + 2);
#pragma unroll
			for (int j = -1; j <= 1; ++j)
			{
				const unsigned char idx = (j + 1) * 3 + (i + 1);
				accX += devGxMask[idx] * buffer[matrixColumn + lRow + j];
				accY += devGyMask[idx] * buffer[matrixColumn + lRow + j];
			}
		}

		const int position = gRow * width + gCol;

		devOutGradients[position] = __fsqrt_rd(__fadd_rd(__fmul_rd(accX, accX), __fmul_rd(accY, accY)));
		devOutDirection[position] = __fmul_rd(__fdiv_rd(fmodf(__fadd_rd(atanf(__fdividef(accY,accX)), M_PI), M_PI), M_PI), 8);
	}
}

__global__ void devNonMaxSuppression(float *nonMaxSupp, float *directions, const unsigned int width, const unsigned int height)
{
	const unsigned int	row = blockIdx.y * blockDim.y + threadIdx.y,
						col = blockIdx.x * blockDim.x + threadIdx.x,
						position = row * (width) + col;

	if (row < height && col < width)
	{
		const float dir = directions[position],
					baseGrad = tex2D(devImageTextureFloat, col, row);

		if (((dir <= 1 || dir > 7) && baseGrad > tex2D(devImageTextureFloat, col - 1, row) &&
			 baseGrad > tex2D(devImageTextureFloat, col + 1, row)) || // 0 deg
			((dir > 1 && dir <= 3) && baseGrad > tex2D(devImageTextureFloat, col + 1, row - 1) &&
			 baseGrad > tex2D(devImageTextureFloat, col - 1, row + 1)) || // 45 deg
			((dir > 3 && dir <= 5) && baseGrad > tex2D(devImageTextureFloat, col, row - 1) &&
			 baseGrad > tex2D(devImageTextureFloat, col, row + 1)) || // 90 deg
			((dir > 5 && dir <= 7) && baseGrad > tex2D(devImageTextureFloat, col - 1, row - 1) &&
			 baseGrad > tex2D(devImageTextureFloat, col + 1, row + 1)))   // 135 deg
			nonMaxSupp[position] = baseGrad;
		else
			nonMaxSupp[position] = 0;

	}
}

void CUDAGauss(unsigned char *devOut, const unsigned int width, const unsigned int height)
{
	dim3 dimGrid(ceil(width / BLOCK_SIZE_1D), ceil(height / BLOCK_SIZE_1D));
	dim3 dimBlock(BLOCK_SIZE_1D, BLOCK_SIZE_1D);

	devGaussKernel<<<dimGrid, dimBlock>>>(devOut, width, height);
}


void CUDAGradients(float *devOutGradients, float *devOutDirection, const unsigned int width, const unsigned int height)
{
	dim3 dimGrid(ceil(width / BLOCK_SIZE_1D), ceil(height / BLOCK_SIZE_1D));
	dim3 dimBlock(BLOCK_SIZE_1D, BLOCK_SIZE_1D);

	devGradients <<<dimGrid, dimBlock>>>(devOutGradients, devOutDirection, width, height);
}


void CUDANonMaximalSuppresion(float *devNonMaxSup, float *devDirections, const unsigned int width, const unsigned int height)
{
	dim3 dimGrid(ceil(width / BLOCK_SIZE_1D), ceil(height / BLOCK_SIZE_1D));
	dim3 dimBlock(BLOCK_SIZE_1D, BLOCK_SIZE_1D);

	devNonMaxSuppression <<<dimGrid, dimBlock>>>(devNonMaxSup, devDirections, width, height);
}

//rebind textury
void CUDARebindTextureChar(unsigned char *devIn, const unsigned int size, const bool unbind = true)
{
	if (unbind)
		cudaUnbindTexture(devImageTextureChar);

	cudaMemcpyToArray(devImageChar, 0, 0, devIn, size, cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(devImageTextureChar, devImageChar);
}
void CUDARebindTextureFloat(float *devIn, const unsigned int size, const bool unbind = true)
{
	if (unbind)
		cudaUnbindTexture(devImageTextureFloat);

	cudaMemcpyToArray(devImageFloat, 0, 0, devIn, size, cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(devImageTextureFloat, devImageFloat);
}

int main() {
	int width, height;
	unsigned char *image = readBMP("data/lena.bmp", &width, &height);
	unsigned char *gauss = new unsigned char[width * height];
	int *gradientX = new int[width * height];
	int *gradientY = new int[width * height];
	float *gradients = new float[width * height];
	float *directions = new float[width * height];
	float *nonMaxSupp = new float[width * height];
	int tmin = 50, tmax = 60;
	const unsigned int imageSizeBytes = width * height * sizeof(unsigned char);

	clock_t a,b;

	initDev(width, height);

	//load the texture (raw iamge)
	cudaMemcpyToArray(devImageChar, 0, 0, image, imageSizeBytes, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(devImageTextureChar, devImageChar);

	//step 1 - gauss (CUDA)
	unsigned char *devGauss;
	cudaMalloc((void **) &devGauss, imageSizeBytes);

	a = clock();
	CUDAGauss(devGauss, width, height);
	cudaDeviceSynchronize();
	b = clock();
	printf("gauss: %lf\n", double(b-a)/CLOCKS_PER_SEC);

	a = clock();
	//update texture memory (replace raw with gauss)
	CUDARebindTextureChar(devGauss, imageSizeBytes);
	cudaDeviceSynchronize();
	b = clock();
	printf("text rbind: %lf\n", double(b-a)/CLOCKS_PER_SEC);

	float *devGradients, *devDirections;
	cudaMalloc((void **) &devGradients, imageSizeBytes * sizeof(float));
	cudaMalloc((void **) &devDirections, imageSizeBytes * sizeof(float));

	a = clock();
	CUDAGradients(devGradients, devDirections, width, height);
	cudaDeviceSynchronize();
	b = clock();
	printf("gradients: %lf\n", double(b-a)/CLOCKS_PER_SEC);

	a = clock();
	//load the float texture (gradients)
	cudaMemcpyToArray(devImageFloat, 0, 0, devGradients, imageSizeBytes * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(devImageTextureFloat, devImageFloat);
	CUDARebindTextureChar(devGauss, imageSizeBytes);
	cudaDeviceSynchronize();
	b = clock();
	printf("text load (float): %lf\n", double(b-a)/CLOCKS_PER_SEC);

	float *devNonMaxSupp;
	cudaMalloc((void **) &devNonMaxSupp, imageSizeBytes * sizeof(float));

	a = clock();
	CUDANonMaximalSuppresion(devNonMaxSupp, devDirections, width, height);
	cudaDeviceSynchronize();
	b = clock();
	printf("non max sup: %lf\n", double(b-a)/CLOCKS_PER_SEC);

	cudaMemcpy(nonMaxSupp, devNonMaxSupp, imageSizeBytes * sizeof(float), cudaMemcpyDeviceToHost);

	//step 2 - intensity gradient (Sobel)

	//convolutionWide(gauss, gradientX, width, height, Gx, 3);
	//convolutionWide(gauss, gradientY, width, height, Gy, 3);

	// step 3 - non-maximum suppression
/*
	for (int i = 1; i < width - 1; i++) {
		for (int j = 1; j < height - 1; j++) {
			const int c = (width * j) + (i);
			const int nn = c - (width);
			const int ss = c + (width);
			const int ww = c + 1;
			const int ee = c - 1;
			const int nw = nn + 1;
			const int ne = nn - 1;
			const int sw = ss + 1;
			const int se = ss - 1;

			//const float dir = (float) (fmod(atan2(gradientY[c], gradientX[c]) + M_PI, M_PI) / M_PI) * 8;
			const float dir = directions[c];

			if (((dir <= 1 || dir > 7) && gradients[c] > gradients[ee] &&
				 gradients[c] > gradients[ww]) || // 0 deg
				((dir > 1 && dir <= 3) && gradients[c] > gradients[nw] &&
				 gradients[c] > gradients[se]) || // 45 deg
				((dir > 3 && dir <= 5) && gradients[c] > gradients[nn] &&
				 gradients[c] > gradients[ss]) || // 90 deg
				((dir > 5 && dir <= 7) && gradients[c] > gradients[ne] &&
				 gradients[c] > gradients[sw]))   // 135 deg
				nonMaxSupp[c] = gradients[c];
			else
				nonMaxSupp[c] = 0;
		}
	}
*/

	int *edges = new int[width * height];
	unsigned char *out = new unsigned char[width * height];

	// step 4 - Tracing edges with hysteresis
	for (int j = 1; j < height - 1; j++) {
		for (int i = 1; i < width - 1; i++) {
			const int c = (width * j) + (i);
			if (nonMaxSupp[c] >= tmax && out[c] == 0) { // trace edges
				out[c] = 255;
				int nedges = 1;
				edges[0] = c;

				do {
					const int t = edges[--nedges];

					int nbs[8]; // neighbours
					nbs[0] = t - (width);     // nn
					nbs[1] = t + (width);     // ss
					nbs[2] = t + 1;      // ww
					nbs[3] = t - 1;      // ee
					nbs[4] = nbs[0] + 1; // nw
					nbs[5] = nbs[0] - 1; // ne
					nbs[6] = nbs[1] + 1; // sw
					nbs[7] = nbs[1] - 1; // se

					for (int k = 0; k < 8; k++)
						if (nonMaxSupp[nbs[k]] >= tmin && out[nbs[k]] == 0) {
							out[nbs[k]] = 255;
							edges[nedges++] = nbs[k];
						}
				} while (nedges > 0);
			}
		}
	}

	// output the file
	writeBMP("/tmp/copy.bmp", out, width, height);

	return 0;
}
/*


__global__
void invertImage(unsigned char *in, unsigned char *out, int w, int h)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y,
			Col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((Row < h) && (Col < w))
	{
		out[Row * w + Col] = 255 - in[Row * w + Col];
	}
}




int altMain()
{
	int width, height;
	unsigned char *image = readBMP("data/lena.bmp", &width, &height);

	unsigned char *cudaIn, *cudaOut;

	cudaMalloc((void **) &cudaIn, height * width * sizeof(unsigned char));
	cudaMalloc((void **) &cudaOut, height * width * sizeof(unsigned char));
	cudaMemcpy((void **) cudaIn, image, height * width * sizeof(unsigned char), cudaMemcpyHostToDevice);

#define BLOCK_SIZE    32

	dim3 DimGrid(ceil(width / BLOCK_SIZE), ceil(height / BLOCK_SIZE));
	dim3 DimBlock(BLOCK_SIZE, BLOCK_SIZE);

	cudaMemset(cudaOut, 0x00, height * width * sizeof(unsigned char));

	clock_t start, finish;
	double totaltime;
	start = clock();

	invertImage <<<DimGrid, DimBlock>>> (cudaIn, cudaOut, width, height);
	cudaMemcpy(image, cudaOut, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	finish = clock();

	totaltime = (double) (finish - start) / CLOCKS_PER_SEC;

	printf("%f", totaltime);

	cudaFree(cudaOut);
	cudaFree(cudaIn);

	// output the file
	writeBMP("/tmp/copy.bmp", image, width, height);

	return 0;
}
 */