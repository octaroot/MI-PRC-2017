//stary X

//$ nvcc -cubin -arch=sm_50 -Xptxas="-v" main.cu

#include <iostream>
#include <cstring>
#include <cmath>
#include <stdio.h>

#define GAUSS_KERNEL_SUM            159

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

//	cudaReadModeElementType means no conversion on access time (optinally normalized)
texture<unsigned char, 2, cudaReadModeElementType> devImageTextureChar;
texture<float, 2, cudaReadModeElementType> devImageTextureFloat;

// the array bound to the 2D textures above
cudaArray* devImageChar, *devImageFloat;

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
//legacy code, not in use anymore
//legacy code, not in use anymore
//legacy code, not in use anymore

	const unsigned int gRow = (blockIdx.y * blockDim.y) + threadIdx.y,
			gCol = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (gRow < height && gCol < width)
	{
		register unsigned char frame[25];
		frame[0] = tex2D(devImageTextureChar, gCol - 2, gRow - 2);
		frame[1] = tex2D(devImageTextureChar, gCol - 1, gRow - 2);
		frame[2] = tex2D(devImageTextureChar, gCol, gRow - 2);
		frame[3] = tex2D(devImageTextureChar, gCol + 1, gRow - 2);
		frame[4] = tex2D(devImageTextureChar, gCol + 2, gRow - 2);
		frame[5] = tex2D(devImageTextureChar, gCol - 2, gRow - 1);
		frame[6] = tex2D(devImageTextureChar, gCol - 1, gRow - 1);
		frame[7] = tex2D(devImageTextureChar, gCol, gRow - 1);
		frame[8] = tex2D(devImageTextureChar, gCol + 1, gRow - 1);
		frame[9] = tex2D(devImageTextureChar, gCol + 2, gRow - 1);
		frame[10] = tex2D(devImageTextureChar, gCol - 2, gRow);
		frame[11] = tex2D(devImageTextureChar, gCol - 1, gRow);
		frame[12] = tex2D(devImageTextureChar, gCol, gRow);
		frame[13] = tex2D(devImageTextureChar, gCol + 1, gRow);
		frame[14] = tex2D(devImageTextureChar, gCol + 2, gRow);
		frame[15] = tex2D(devImageTextureChar, gCol - 2, gRow + 1);
		frame[16] = tex2D(devImageTextureChar, gCol - 1, gRow + 1);
		frame[17] = tex2D(devImageTextureChar, gCol, gRow + 1);
		frame[18] = tex2D(devImageTextureChar, gCol + 1, gRow + 1);
		frame[19] = tex2D(devImageTextureChar, gCol + 2, gRow + 1);
		frame[20] = tex2D(devImageTextureChar, gCol - 2, gRow + 2);
		frame[21] = tex2D(devImageTextureChar, gCol - 1, gRow + 2);
		frame[22] = tex2D(devImageTextureChar, gCol, gRow + 2);
		frame[23] = tex2D(devImageTextureChar, gCol + 1, gRow + 2);
		frame[24] = tex2D(devImageTextureChar, gCol + 2, gRow + 2);


		const float acc = 2 * frame[0] + 4 * frame[1] + 5 * frame[2] + 4 * frame[3] + 2 * frame[4]
						  + 4 * frame[5] + 9 * frame[6] + 12 * frame[7] + 9 * frame[8] + 4 * frame[9]
						  + 5 * frame[10] + 12 * frame[11] + 15 * frame[12] + 12 * frame[13] + 5 * frame[14]
						  + 4 * frame[15] + 9 * frame[16] + 12 * frame[17] + 9 * frame[18] + 4 * frame[19]
						  + 2 * frame[20] + 4 * frame[21] + 5 * frame[22] + 4 * frame[23] + 2 * frame[24];


		output[gRow * width + gCol] = (unsigned char) __fdiv_rd(acc, GAUSS_KERNEL_SUM);
	}
}



__global__ void devGaussKernelX(float *output, const unsigned int width, const unsigned int height)
{
	register const unsigned int gRow = (blockIdx.y * blockDim.y) + threadIdx.y,
			gCol = (blockIdx.x * blockDim.x) + threadIdx.x;

	//0.113318	0.236003	0.30136	0.236003	0.113318

	//prace s texturou byla stejne dobra, jako bez ni. Na zacatku je tedy zachovano pouziti textur
	if (gRow < height && gCol < width)
	{
		output[gRow * width + gCol] = __fmul_rd(0.113318f, tex2D(devImageTextureChar, gCol - 2, gRow)) +
									  __fmul_rd(0.236003f, tex2D(devImageTextureChar, gCol - 1, gRow)) +
									  __fmul_rd(0.30136f, tex2D(devImageTextureChar, gCol, gRow)) +
									  __fmul_rd(0.236003f, tex2D(devImageTextureChar, gCol + 1, gRow)) +
									  __fmul_rd(0.113318f, tex2D(devImageTextureChar, gCol + 2, gRow));
	}
}

__global__ void devGaussKernelY(float *input, unsigned char *output, const unsigned int width, const unsigned int height)
{
	register const unsigned int gRow = (blockIdx.y * blockDim.y) + threadIdx.y,
			gCol = (blockIdx.x * blockDim.x) + threadIdx.x;

	__shared__ float cache[BLOCK_SIZE_1D + 4][BLOCK_SIZE_1D];

	//mapovani pole [y][x] bylo lepsi nez [x][y]
	const register float center = cache[2 + threadIdx.y][threadIdx.x] = input[gRow * width + gCol];

	//verze, kde nacitat budou 2 vlakna po jednom prvku byla horsi
	if(threadIdx.y == 0)
	{
		cache[threadIdx.y][threadIdx.x] = input[(gRow - ((blockIdx.y == 0) ? 0 : 2)) * width + gCol];
		cache[threadIdx.y + 1][threadIdx.x] = input[(gRow - ((blockIdx.y == 0) ? 0 : 1)) * width + gCol];
	}
	else if (threadIdx.y == BLOCK_SIZE_1D - 1)
	{
		cache[threadIdx.y + 3][threadIdx.x] = input[(gRow + (gRow == height - 1 ? 0 : 1)) * width + gCol];
		cache[threadIdx.y + 4][threadIdx.x] = input[(gRow + (gRow == height - 1 ? 0 : 2)) * width + gCol];
	}

	__syncthreads();

	if (gRow < height && gCol < width)
	{
		//__float2uint_rd nedela zmenu
		output[gRow * width + gCol] = (unsigned char) (__fmul_rd(0.113318f, cache[threadIdx.y][threadIdx.x]) +
													   __fmul_rd(0.236003f, cache[threadIdx.y + 1][threadIdx.x]) +
													   __fmul_rd(0.30136f, center) +
													   __fmul_rd(0.236003f, cache[threadIdx.y + 3][threadIdx.x]) +
													   __fmul_rd(0.113318f, cache[threadIdx.y + 4][threadIdx.x]));
	}
}

__global__ void devGradients(float *devOutGradients, float *devOutDirection, const unsigned int width, const unsigned int height)
{
	register const unsigned int	gRow = (blockIdx.y * blockDim.y) + threadIdx.y,
			gCol = (blockIdx.x * blockDim.x) + threadIdx.x;

	register unsigned char frame[9];
	frame[0] = tex2D(devImageTextureChar, gCol - 1, gRow - 1);
	frame[1] = tex2D(devImageTextureChar, gCol, gRow - 1);
	frame[2] = tex2D(devImageTextureChar, gCol + 1, gRow - 1);
	frame[3] = tex2D(devImageTextureChar, gCol - 1, gRow);
	frame[4] = tex2D(devImageTextureChar, gCol, gRow);
	frame[5] = tex2D(devImageTextureChar, gCol + 1, gRow);
	frame[6] = tex2D(devImageTextureChar, gCol - 1, gRow + 1);
	frame[7] = tex2D(devImageTextureChar, gCol, gRow + 1);
	frame[8] = tex2D(devImageTextureChar, gCol + 1, gRow + 1);

	if (gRow < height && gCol < width)
	{
		//zde deklarace register nepomohla s vykonem
		const int	accX = -frame[0] + frame[2] - frame[3]- frame[3] + frame[5] + frame[5] - frame[6] + frame[8],
				accY = frame[0] + frame[1] + frame[1] + frame[2] - frame[6] - frame[7] - frame[7] - frame[8];

		//rychlejsi hypot nez user defined s pomoci intristics
		devOutGradients[(gRow * width) + gCol] = hypotf(accY, accX);
		devOutDirection[(gRow * width) + gCol] = __fmul_rd(__fdiv_rd(fmodf(__fadd_rd(atanf(__fdividef(accY, accX)), M_PI), M_PI), M_PI), 8);
	}
}

__global__ void devNonMaxSuppression(float *nonMaxSupp, float *directions, const unsigned int width, const unsigned int height)
{
	register const unsigned int	row = blockIdx.y * blockDim.y + threadIdx.y,
			col = blockIdx.x * blockDim.x + threadIdx.x,
			position = row * (width) + col;

	if (row < height && col < width)
	{
		register const float dir = directions[position],
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

void CUDAGauss(float *tmp, unsigned char *devOut, const unsigned int width, const unsigned int height)
{
	dim3 dimGrid(ceil(width / BLOCK_SIZE_1D), ceil(height / BLOCK_SIZE_1D));
	dim3 dimBlock(BLOCK_SIZE_1D, BLOCK_SIZE_1D);

	//devGaussKernel<<<dimGrid, dimBlock>>>(devOut, width, height);

	devGaussKernelX<<<dimGrid, dimBlock>>>(tmp, width, height);
	devGaussKernelY<<<dimGrid, dimBlock>>>(tmp, devOut, width, height);
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
void CUDARebindTextureChar(unsigned char *devIn, const unsigned int size)
{
	cudaUnbindTexture(devImageTextureChar);
	cudaMemcpyToArrayAsync(devImageChar, 0, 0, devIn, size, cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(devImageTextureChar, devImageChar);
}

int main(int argc, const char ** argv) {
	if (argc!=2 && argc!=3)
	{
		printf ("%s [input BMP] [output BMP - optional, defaults to /tmp/copy.bmp]\n", argv[0]);
		return 1;
	}

	int width, height;
	unsigned char *image = readBMP(argv[1], &width, &height);
	int *edges = new int[width * height];
	unsigned char *out = new unsigned char[width * height];
	float *nonMaxSupp = new float[width * height];
	int tmin = 50, tmax = 60;
	const unsigned int imageSizeBytes = width * height * sizeof(unsigned char);

	clock_t total_a,a,b,total_b;
	total_a = clock();
	double goodtime = 0;

	initDev(width, height);

	//load the texture (raw iamge)d
	cudaMemcpyToArray(devImageChar, 0, 0, image, imageSizeBytes, cudaMemcpyHostToDevice);
	cudaBindTextureToArray(devImageTextureChar, devImageChar);

	//step 1 - gauss (CUDA)
	unsigned char *devGauss;
	cudaMalloc((void **) &devGauss, imageSizeBytes);

	float *devGradients, *devDirections;
	cudaMalloc((void **) &devGradients, imageSizeBytes * sizeof(float));
	cudaMalloc((void **) &devDirections, imageSizeBytes * sizeof(float));

	//taktez slouzi jako temp uloziste pro gausse
	float *devNonMaxSupp;
	cudaMalloc((void **) &devNonMaxSupp, imageSizeBytes * sizeof(float));

	a = clock();
	CUDAGauss(devNonMaxSupp, devGauss, width, height);
	cudaDeviceSynchronize();
	b = clock();
	printf("gauss: %lf\n", double(b-a)/CLOCKS_PER_SEC);
	goodtime += double(b-a);

	a = clock();
	//update texture memory (replace raw with gauss)
	CUDARebindTextureChar(devGauss, imageSizeBytes);
	cudaDeviceSynchronize();
	b = clock();
	printf("text rbind: %lf\n", double(b-a)/CLOCKS_PER_SEC);
	goodtime += double(b-a);


	a = clock();
	CUDAGradients(devGradients, devDirections, width, height);
	cudaDeviceSynchronize();
	b = clock();
	printf("gradients: %lf\n", double(b-a)/CLOCKS_PER_SEC);
	goodtime += double(b-a);

	a = clock();
	//load the float texture (gradients)
	cudaMemcpyToArrayAsync(devImageFloat, 0, 0, devGradients, imageSizeBytes * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(devImageTextureFloat, devImageFloat);
	cudaDeviceSynchronize();
	b = clock();
	printf("text load (float): %lf\n", double(b-a)/CLOCKS_PER_SEC);
	goodtime += double(b-a);

	a = clock();
	CUDANonMaximalSuppresion(devNonMaxSupp, devDirections, width, height);
	cudaDeviceSynchronize();
	b = clock();
	printf("NMS: %lf\n", double(b-a)/CLOCKS_PER_SEC);
	goodtime += double(b-a);

	a = clock();
	cudaMemcpy(nonMaxSupp, devNonMaxSupp, imageSizeBytes * sizeof(float), cudaMemcpyDeviceToHost);
	b = clock();
	printf("memcpy to host: %lf\n", double(b-a)/CLOCKS_PER_SEC);
	goodtime += double(b-a);

	a = clock();
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
	b = clock();
	printf("hystersis: %lf\n", double(b-a)/CLOCKS_PER_SEC);

	// output the file
	writeBMP(argc == 3 ? argv[2] : "/tmp/copy.bmp", out, width, height);

	total_b = clock();
	printf("GPU TIME: %lf\n", goodtime/CLOCKS_PER_SEC);
	goodtime += double(b-a);
	printf("GPU+CPU(hystersis) TIME: %lf\n", goodtime/CLOCKS_PER_SEC);
	printf("EXEC TIME: %lf\n", double(total_b-total_a)/CLOCKS_PER_SEC);


	return 0;
}