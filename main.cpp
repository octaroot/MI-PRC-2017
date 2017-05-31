#include <iostream>
#include <cstring>

#define KERNEL_SIZE	159

// via http://stackoverflow.com/questions/9296059/read-pixel-value-in-bmp-file
unsigned char *readBMP(const char *filename, int *width, int *height)
{
	FILE *f = fopen(filename, "rb");
	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	*width = *(int *) &info[18];
	*height = *(int *) &info[22];

	int size = 3 * (*width) * (*height);
	unsigned char *data = new unsigned char[size]; // allocate 3 bytes per pixel
	fread(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);


	//endianity (BGR -> RGB)
	for (int i = 0; i < size; i += 3)
	{
		unsigned char tmp = data[i];
		data[i] = data[i + 2];
		data[i + 2] = tmp;
	}

	return data;
}

void writeBMP(const char * filename, unsigned char * data, int width, int height)
{
	int size = 3 * width * height;

	//endianity (BGR -> RGB)
	for (int i = 0; i < size; i += 3)
	{
		unsigned char tmp = data[i];
		data[i] = data[i + 2];
		data[i + 2] = tmp;
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
	fwrite(data, sizeof(unsigned char), size, f); // read the rest of the data at once
	fclose(f);
}

int main()
{
	int width, height;
	unsigned char *image = readBMP("/home/martin/Projects/mi-prc-cannyedge/data/lena.bmp", &width, &height);
	unsigned char *aux = new unsigned char[width * height * 3];
	//The color of pixel (i, j) is stored at data[j * width + i], data[j * width + i + 1] and data[j * width + i + 2].

	memcpy(aux, image, width * height * 3);

	//step 1 - gauss, |kernel|=5, sum=159
	const char kernel[5][5]{{2, 4,  5,  4,  2},
							{4, 9,  12, 9,  4},
							{5, 12, 15, 12, 5},
							{4, 9,  12, 9,  4},
							{2, 4,  5,  4,  2},};

	for (int x = 2; x < width - 3; ++x)
	{
		for (int y = 2; y < height - 3; ++y)
		{
			for (int color = 0; color < 3; ++color)
			{
				int newTotal = 0;
				for (int i = 0; i < 5; ++i)
				{
					for (int j = 0; j < 5; ++j)
					{
						newTotal += image[(y + j - 2) * (3*height) + 3*(x + i - 2) + color] * kernel[i][j];
					}
				}
				aux[y * (3*height) + 3*x + color] = (unsigned char) (newTotal / KERNEL_SIZE);
			}
		}
	}


	writeBMP("/tmp/copy.bmp", aux, width, height);
	return 0;
}