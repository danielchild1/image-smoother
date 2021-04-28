#include <Kokkos_Core.hpp>
#include <fstream>
//#include <filesystem>
#include <cstdio>
#include <string>

using std::string;
using std::ios;
using std::ifstream;
using std::ofstream;

double duration(std::chrono::high_resolution_clock::time_pointstart, std::chrono::high_resolution_clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {

	Kokkos::initialize(argc, argv);
	{
		string filename = "/share/HK-7_left_H6D-400c-MS.bmp";
		//std::uintmax_t filesize = std::filesystem::file_size(filename);
		//printf("The file size is %ju\n", filesize);
		// Open File
		ifstream fin(filename, ios::in | ios::binary);

		if(!fin.is_open()) {
			printf("File not opened\n");
			return -1;
		}// The first 14 bytes are the header, containing four values.  Get those four values.

		char header[2];
		uint32_t filesize;
		uint32_t dummy;
		uint32_t offset;
		fin.read(header, 2);
		fin.read((char*)&filesize, 4);
		fin.read((char*)&dummy, 4);
		fin.read((char*)&offset, 4);
		printf("header: %c%c\n", header[0], header[1]);
		printf("filesize: %u\n", filesize);
		printf("dummy %u\n", dummy);
		printf("offset: %u\n", offset);
		int32_t sizeOfHeader;
		int32_t width;
		int32_t height;
		fin.read((char*)&sizeOfHeader, 4);
		fin.read((char*)&width, 4);
		fin.read((char*)&height, 4);
		printf("The width: %d\n", width);
		printf("The height: %d\n", height);
		uint16_t numColorPanes;
		uint16_t numBitsPerPixel;
		fin.read((char*)&numColorPanes, 2);
		fin.read((char*)&numBitsPerPixel, 2);
		printf("The number of bits per pixel: %u\n", numBitsPerPixel);
		if (numBitsPerPixel == 24) {
			printf("This bitmap uses rgb, where the first byte is blue, second byte is green, third byte is red.\n");
		}
		uint32_t rowSize = (numBitsPerPixel * width + 31) / 32 * 4;
		//printf("Each row in the image requires %u bytes\n", rowSize);

		// Jump to offset where the bitmap pixel data starts
		fin.seekg(offset, ios::beg);


		// Read the data part of the file
		unsigned char* h_buffer = new unsigned char[filesize-offset];
		fin.read((char*)h_buffer, filesize-offset);
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point end;

		printf("The first pixel is located in the bottom left.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[0], h_buffer[1], h_buffer[2]);
		printf("The second pixel is to the right.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[3], h_buffer[4], h_buffer[5]);

		// TODO: Read the image into Kokkos views 
		Kokkos::View<char**, Kokkos::LayoutRight> inputImage("inputImage", rowSize, height);
		Kokkos::View<char**, Kokkos::LayoutRight> outputImage("outputImage", rowSize, height);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < rowSize; j++) {
				char temp;
				fin.read((char*)&temp, 1);
				inputImage(i, j) = temp;
			}
		}

		// i/j with height/width.
		delete[] h_buffer;
		// BLUE GREEN RED
		start = std::chrono::high_resolution_clock::now();
		// TODO: Perform the blurring
		Kokkos::parallel_for(Kokkos::RangePolicy<>(0, rowSize * height),
			KOKKOS_LAMBDA(const int n) {
			int i = n / height;
			int j = n % rowSize;
			if (i >= 3 && j >= 3 && j < width-3 && i < height-3) {
				char storagePlace = inputImage(i, j);
				storagePlace *= 41;
				//26
				storagePlace += (inputImage(i - 1, j) * 26 + inputImage(i + 1, j) * 26 + inputImage(i, j + 3) * 26 + inputImage(i, j - 3) * 26);
				//16
				storagePlace += (inputImage(i + 1, j + 3) * 16 + inputImage(i - 1, j + 3) * 16 + inputImage(i - 1, j - 3) * 16 + inputImage(i - 1, j + 3) * 16);
				//7
				storagePlace += (inputImage(i - 2, j) * 7 + inputImage(i + 2, j) * 7 + inputImage(i, j + 6) * 7 + inputImage(i, j - 6) * 7);
				//4
				storagePlace += (inputImage(i + 2, j + 3) * 4 + inputImage(i + 1, j + 6) * 4 + inputImage(i - 1, j + 6) * 4 + inputImage(i + 2, j + 3) * 4 + inputImage(i + 2, j - 3) * 4 + inputImage(i + 1, j - 6) * 4 + inputImage(i - 1, j - 6) * 4 + inputImage(i - 2, j - 3) * 4);
				//1
				storagePlace += (inputImage(i - 2, j + 6) + inputImage(i + 2, j + 6) + inputImage(i + 2, j - 6) + +inputImage(i - 2, j - 6));

				outputImage(i,j) = storagePlace/271;
			}

		}
			
			)

		end = std::chrono::high_resolution_clock::now();printf("Time -%g ms\n", duration(start,end));

		// TODO: Verification
		printf("The red, green, blue at (8353, 9111) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (8351, 9113) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (6352, 15231) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (10559, 10611) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);
		printf("The red, green, blue at (10818, 20226) (origin bottom left) is (%d, %d, %d)\n", 0, 0, 0);

		//print out to file output.bmp
		string outputFile = "output.bmp";
		ofstream fout;
		fout.open(outputFile, ios::binary);

		//Copy of the old heaers into the new output
		fin.seekg(0, ios::beg);
		//read the data part of the file
		char* headers = new char[offset];
		fin.read(headers, offset);
		fout.seekp(0, ios::beg);
		fout.write(headers, offset);
		delete[] headers;

		fout.seekp(offset, ios::beg);
		//TODO: Copy out the rest of the view to file (hint, use fout.put())
		for (int c = 0; c < width; c++) {
			for (int r = 0; r < height; r++) {
				fout.put(outputFile(r, c));
			}
		}
		
		fout.close();
	}
	Kokkos::finalize();
}