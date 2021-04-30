#include <Kokkos_Core.hpp>
#include <fstream>
#include <chrono>
//#include <filesystem>
#include <cstdio>
#include <string>

using std::string;
using std::ios;
using std::ifstream;
using std::ofstream;

double duration(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {

	Kokkos::initialize(argc, argv);
	{
		string filename = "/share/HK-7_left_H6D-400c-MS_screw.bmp";
		//std::uintmax_t filesize = std::filesystem::file_size(filename);
		//printf("The file size is %ju\n", filesize);
		// Open File
		ifstream fin(filename, ios::in | ios::binary);

		if (!fin.is_open()) {
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
		unsigned char* h_buffer = new unsigned char[filesize - offset];
		fin.read((char*)h_buffer, filesize - offset);
		std::chrono::high_resolution_clock::time_point start;
		std::chrono::high_resolution_clock::time_point end;

		printf("The first pixel is located in the bottom left.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[0], h_buffer[1], h_buffer[2]);
		printf("The second pixel is to the right.  Its blue/green/red values are (%u, %u, %u)\n", h_buffer[3], h_buffer[4], h_buffer[5]);

		// TODO: Read the image into Kokkos views 
		Kokkos::View<char**, Kokkos::LayoutRight> inputImage("inputImage", height, rowSize);
		Kokkos::View<char**, Kokkos::LayoutRight> outputImage("outputImage", height, rowSize);

		Kokkos::View<char**, Kokkos::LayoutRight>::HostMirror hostIn = create_mirror(inputImage);
		Kokkos::View<char**, Kokkos::LayoutRight>::HostMirror hostOut = create_mirror(outputImage);
		printf("location a\n");
		for (int i = 0; i < height * rowSize; i++) {
			int c = i % rowSize;
			int r = i / rowSize;
			hostIn(r, c) = h_buffer[(r * rowSize) + c];
		}

		Kokkos::deep_copy(inputImage, hostIn);

		printf("location b\n");
		// i/j with height/width.
		delete[] h_buffer;
		// BLUE GREEN RED
		start = std::chrono::high_resolution_clock::now();
		// TODO: Perform the blurring
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0, rowSize * height),
			KOKKOS_LAMBDA(const int n) {
			int j = n % rowSize; //col
			int i = n / rowSize; //row

			
			
			if (i < 3 ||  i > (height - 4) || j < 7 || j > (rowSize - 8) ) {
				outputImage(i, j) = inputImage(i, j);
				
			}
			else {
			int32_t storagePlace = inputImage(i, j);
				storagePlace *= 41;
				//26
				storagePlace += (26 * (inputImage(i - 1, j)  + inputImage(i + 1, j) + inputImage(i, j + 3) + inputImage(i, j - 3)));
				//16
				storagePlace += (16 * (inputImage(i + 1, j + 3) + inputImage(i + 1, j - 3) + inputImage(i - 1, j - 3)  + inputImage(i - 1, j + 3) ));
				//7
				storagePlace += (7 * (inputImage(i - 2, j) + inputImage(i + 2, j) + inputImage(i, j + 6) + inputImage(i, j - 6)));
				//4
				storagePlace += (4 * (inputImage(i + 2, j + 3) + inputImage(i + 1, j + 6) + inputImage(i - 1, j + 6) + inputImage(i - 2, j + 3) + inputImage(i + 2, j - 3) + inputImage(i + 1, j - 6) + inputImage(i - 1, j - 6) + inputImage(i - 2, j - 3)));
				//1
				storagePlace += (inputImage(i - 2, j + 6) + inputImage(i + 2, j + 6) + inputImage(i + 2, j - 6) + inputImage(i - 2, j - 6));

				outputImage(i, j) = storagePlace / 273;
			}
			

		}

		);
		// printf("does: %d, doesNot: %d\n", does, doesNot);
		printf("location c\n");

		end = std::chrono::high_resolution_clock::now();


		printf("Time %g ms\n", duration(start, end));

		Kokkos::deep_copy(hostOut, outputImage);

		// TODO: Verification
		// BLUE GREEN RED
		// printf("Did not make it to the print out\n");
		// printf("The red, green, blue at (8353, 9111) (origin bottom left) is (%d, %d, %d)\n", hostOut(8353, (9111 * 3) + 2), hostOut(8353, (9111 * 3) + 1), hostOut(8353, (9111 * 3)));
		// printf("The red, green, blue at (8351, 9113) (origin bottom left) is (%d, %d, %d)\n", hostOut(8351, 27333 + 2), hostOut(8351, 27333 + 1), hostOut(8351, 27333));
		// printf("The red, green, blue at (6352, 15231) (origin bottom left) is (%d, %d, %d)\n", hostOut(6352, 45963 + 2), hostOut(6352, 45963 + 1), hostOut(6352, 45963));
		// printf("The red, green, blue at (10559, 10611) (origin bottom left) is (%d, %d, %d)\n", hostOut(10559, 31833 + 2), hostOut(10559, 31833 + 1), hostOut(10559, 31833));
		// printf("The red, green, blue at (10818, 20226) (origin bottom left) is (%d, %d, %d)\n", hostOut(10818, 60678 + 2), hostOut(10818, 60678 + 1), hostOut(10818, 60678));

		//print out to file output.bmp
		string outputFile = "screw-dan.bmp";
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
		for (int i = 0; i < height * rowSize; i++) {
			int c = i % rowSize;
			int r = i / rowSize;
			fout.put((char)hostOut(r, c));
		}
		printf("Location D\n");

		fout.close();
	}
	Kokkos::finalize();
}