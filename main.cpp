#include <iostream>
#include <cstring>
#include <cmath>

#define KERNEL_SIZE            159

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

int main() {
	int width, height;
	unsigned char *image = readBMP("data/lena.bmp", &width, &height);
	unsigned char *gauss = new unsigned char[width * height];
	int *gradientX = new int[width * height];
	int *gradientY = new int[width * height];
	double *gradients = new double[width * height];
	double *nonMaxSupp = new double[width * height];
	//The color of pixel (i, j) is stored at data[j * width + i], data[j * width + i + 1] and data[j * width + i + 2].

	int tmin = 50, tmax = 60;

	memcpy(gauss, image, width * height);

	//step 1 - gauss, |kernel|=5, sum=159
	const char kernel[]{2, 4, 5, 4, 2,
						4, 9, 12, 9, 4,
						5, 12, 15, 12, 5,
						4, 9, 12, 9, 4,
						2, 4, 5, 4, 2};
	convolution(image, gauss, width, height, kernel, 5, KERNEL_SIZE);

	//step 2 - intensity gradient (Sobel)
	const char Gx[] = {-1, 0, 1,
					   -2, 0, 2,
					   -1, 0, 1};

	convolutionWide(gauss, gradientX, width, height, Gx, 3);

	const char Gy[] = {1, 2, 1,
					   0, 0, 0,
					   -1, -2, -1};

	convolutionWide(gauss, gradientY, width, height, Gy, 3);

	//step 3 - hystersis
	for (int i = 1; i < width - 1; i++) {
		for (int j = 1; j < height - 1; j++) {
			const int c = width * j + i;
			gradients[c] = (float) hypot(gradientX[c], gradientY[c]);
		}
	}

	// Non-maximum suppression, straightforward implementation.
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

			const float dir = (float) (fmod(atan2(gradientY[c], gradientX[c]) + M_PI, M_PI) / M_PI) * 8;

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

	int *edges = new int[width * height];
	unsigned char *out = new unsigned char[width * height];

	// Tracing edges with hysteresis
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
	writeBMP("/tmp/copy.bmp", out, width, height);

	return 0;
}