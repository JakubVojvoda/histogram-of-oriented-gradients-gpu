/**
 *  Histogram of oriented gradient on GPU 
 *  by Jakub Vojvoda, vojvoda@swdeveloper.sk
 *  2015  
 *
 *  Implementation based on the original publication of Dalal and Triggs
 *  https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
 *
 *  licence: GNU LGPL v3
 *  file: hog.cl
 */

#ifdef cl_khr_fp64
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define HISTOGRAM_BINS 9
#define CELL_WIDTH  8
#define CELL_HEIGHT 8
#define BLOCK_WIDTH  16
#define BLOCK_HEIGHT 16

#define CELLS_IN_BLOCK  ((BLOCK_WIDTH/CELL_WIDTH) * (BLOCK_HEIGHT/CELL_HEIGHT))
#define DESCRIPTOR_SIZE (CELLS_IN_BLOCK * HISTOGRAM_BINS)

#define WEIGHTED   1
#define NORMALIZED 1
#define PREDICTION 1

#ifndef M_PI
  #define M_PI 3.14159265f
#endif

// Local atomic addition of floating point values based on tutorial Atomic 
// operations and floating point numbers in OpenCL: http://simpleopencl.blogspot.com
// 
void atomic_add_local(volatile __local float *source, const float operand) 
{    
  // replaces type casting of pointers
  union {
    unsigned int vint;
    float vfloat; 
  } newval;
 
  // replaces type casting of pointers
  union {
    unsigned int vint;
    float vfloat;
  } prevval;
 
  // serialization of the memory access
  do {
    prevval.vfloat = *source;
    newval.vfloat = prevval.vfloat + operand;
  } while (atom_cmpxchg((volatile local unsigned int *)source, prevval.vint, newval.vint) != prevval.vint);
}

// Gradient computation from input image 
// the magnitude is stored in *g_magn and the angle is in *g_angle
//
__kernel void compute_gradient(__global uchar *img, __global float *g_magn, __global float *g_angle, int width, int height) 
{ 
  // global index 
  int global_x = (int) get_global_id(0);
  int global_y = (int) get_global_id(1);

  if (global_x >= width || global_y >= height) { 
    return;
  }
	
  // assignment of zero value to borders
  if (global_x == 0 || global_y == 0 || global_x == width - 1 || global_y == height - 1) { 
    g_magn[global_x + global_y * width] = 0;
    g_angle[global_x + global_y * width] = 0;
    return;
  }

  // computation of vertical and horizontal gradient
  float vert  = img[(global_x + 1) + global_y * width] - img[(global_x - 1) + global_y * width];
  float horiz = img[global_x + (global_y + 1) * width] - img[global_x + (global_y - 1) * width];
	
  // magnitude computation	
  g_magn[global_x + global_y * width]  = sqrt(vert * vert + horiz * horiz);

  // computation of angle in degrees
  float angle = (horiz != 0) ? atan(vert / horiz) * 180.0 / M_PI : 0;
  g_angle[global_x + global_y * width] = (angle < 0) ? angle + 180 : angle;
}

// HOG features computation in single window of the input image
// window descriptor is stored in global memory *descriptor
//  
__kernel void compute_window_descriptor(
  __read_only __global uchar *image, 
  __read_only int width, 
  __read_only int height, 
  __write_only __global float *descriptor)
{ 
  // global index
  int global_x = get_global_id(0);
  int global_y = get_global_id(1);
  
  // group index 
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);
  
  // local index 
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);
   
  if (global_x >= width || global_y >= height) { 
    return;
  }

  // local memory declaration
  __local float features[DESCRIPTOR_SIZE];
		
  int cell_index = 0;
  int hist_index = 0;
 
  if (local_x >= BLOCK_WIDTH || local_y >= BLOCK_HEIGHT) { 
    return;
  }
	
  // block index within window
  cell_index = (int)(local_x / CELL_WIDTH) + (BLOCK_WIDTH / CELL_WIDTH) * ((int)(local_y / CELL_HEIGHT));

  // initialization of the block descriptor (local memory) 
  for (int i = 0; i < DESCRIPTOR_SIZE; i++) { 
    features[i] = 0;
  }	

  barrier(CLK_LOCAL_MEM_FENCE);

  float magnitude = 0; 
  float angle = 0;

  // computation of magnitude and angle of gradient
  if (global_x > 0 && global_x < width - 1 && global_y > 0 && global_y < height - 1) { 
		
    float vert  = image[(global_x + 1) + global_y * width] - image[(global_x - 1) + global_y * width];
    float horiz = image[global_x + (global_y + 1) * width] - image[global_x + (global_y - 1) * width];
		
    magnitude  = sqrt(vert * vert + horiz * horiz);
		
    float signed_angle = (horiz != 0) ? atan(vert / horiz) * 180.0 / M_PI : 0;
    angle = (signed_angle < 0) ? signed_angle + 180 : signed_angle;
  }

#if WEIGHTED
	
  // weighted histogram calculation 
  hist_index = floor((angle + 10.0) / 20.0) - 1;
  int index = hist_index + cell_index * HISTOGRAM_BINS;

  if (hist_index < 0) { 
    atomic_add_local(&(features[0]), magnitude);
  }
  else if (hist_index > HISTOGRAM_BINS - 2) { 
    atomic_add_local(&(features[HISTOGRAM_BINS - 1]), magnitude);
  else { 
    float lvalue = hist_index * 20.0 + 10.0;
    float rvalue = (hist_index + 1) * 20.0 + 10.0;
		
    float lweight = 1 - (fabs(lvalue - angle) / 20.0);
    float rweight = 1 - (fabs(rvalue - angle) / 20.0);
		
    atomic_add_local(&(features[index]), lweight * magnitude);
    atomic_add_local(&(features[index + 1]), rweight * magnitude); 
  }

#else
	
  // unweighted histogram calculation
  hist_index = floor(angle / 20.0);
  hist_index = (hist_index > 8) ? 8 : hist_index;

  int index = hist_index + cell_index * HISTOGRAM_BINS;
  
  atomic_add_local(&(features[index]), magnitude);
	
#endif

  barrier(CLK_LOCAL_MEM_FENCE);

  float norm = 1.0;

#if NORMALIZED
	
  // block normalization
  int local_index = local_x + local_y * (int)get_local_size(0);
	
  __local float tmp[DESCRIPTOR_SIZE];

  if (local_index < DESCRIPTOR_SIZE) {
    tmp[local_index] = features[local_index] * features[local_index]; 
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // parallel reduction 
  for (int i = DESCRIPTOR_SIZE >> 1; i > 0; i >>= 1) { 
		
    if (local_index < i) { 
      tmp[local_index] += tmp[local_index + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_index == 0) {
    norm = sqrt(tmp[0]);
  }

#endif	
	
  // result writing to global memory on specific index
  if (local_x == 0 && local_y == 0) { 
		
    int position = DESCRIPTOR_SIZE * (group_x + group_y * (width / BLOCK_WIDTH));
	 
    for (int i = 0; i < DESCRIPTOR_SIZE; i++) { 
      descriptor[position + i] = features[i] / norm;
    }
  }

}

// Predicts the response for input samples *descriptor
// based on weights *y and bias b
//
__kernel void predict(
  __read_only __global float *descriptor,
  __read_only int size,
  __read_only __global float *y,
  __read_only float b,
  __local float *tmp,
  __write_only __global float *prediction)
{ 
  // global index
  int index = get_global_id(0);

  if (index >= size) { 
    return;
  }

  // product of samples and weights
  tmp[index] = descriptor[index] * y[index];
	
  barrier(CLK_LOCAL_MEM_FENCE);

  // parallel reduction
  for (int i = size >> 1; i > 0; i >>= 1) { 
	
    if (index < i) { 
      tmp[index] += tmp[index + i];
    }
		
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // sign based decision
  if (index == 0) {
    *prediction = ((tmp[0] + b) < 0) ? -1.0f : 1.0f;
  }
}

// HOG descriptor calculation from input image
// prediction for each window is stored in *prediction 
//
__kernel void compute_hog_descriptor(
  __read_only __global  uchar *image, 
  __read_only int image_width, 
  __read_only int image_height, 
  __read_only int win_width, 
  __read_only int win_height, 
  __write_only __global float *descriptor,
  __read_only __global float *y,
  __read_only float b,
  __write_only __global float *prediction)
{ 
  // global index 
  int global_x = get_global_id(0);
  int global_y = get_global_id(1);

  // group index
  int group_x = get_group_id(0);
  int group_y = get_group_id(1);

  // local index 
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);

  if (global_x >= image_width || global_y >= image_height) { 
    return;
  }

  // window index 
  int win_x = global_x / win_width;
  int win_y = global_y / win_height;
			
  if (local_x >= BLOCK_WIDTH || local_y >= BLOCK_HEIGHT) { 
    return;
  }

  // block features declaration and initialization
  __local float features[DESCRIPTOR_SIZE];

  for (int i = 0; i < DESCRIPTOR_SIZE; i++) { 
    features[i] = 0;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // computatation of cell index within window 
  int cell_index = (int)(local_x / CELL_WIDTH) + (BLOCK_WIDTH / CELL_WIDTH) * ((int)(local_y / CELL_HEIGHT));

  float magnitude = 0;
  float angle = 0;

  // computation of magnitude and angle of gradient
  if (global_x > 0 && global_x < image_width - 1 && global_y > 0 && global_y < image_height - 1) { 
	
    float vert  = image[(global_x + 1) + global_y * image_width] - image[(global_x - 1) + global_y * image_width];
    float horiz = image[global_x + (global_y + 1) * image_width] - image[global_x + (global_y - 1) * image_width];
	
    magnitude  = sqrt(vert * vert + horiz * horiz);
		
    float signed_angle = (horiz != 0) ? atan(vert / horiz) * 180.0 / M_PI : 0;
    angle = (signed_angle < 0) ? signed_angle + 180 : signed_angle;
  }

#if WEIGTHED 
	
  // weighted histogram calculation 
  int hist_index = floor((angle + 10.0) / 20.0) - 1;
  int index = hist_index + cell_index * HISTOGRAM_BINS;

  if (hist_index < 0) { 
    atomic_add_local(&(features[0]), magnitude);
  }
  else if (hist_index > HISTOGRAM_BINS - 2) { 
    atomic_add_local(&(features[HISTOGRAM_BINS - 1]), magnitude);
  }
  else { 
    float lvalue = hist_index * 20.0 + 10.0;
    float rvalue = (hist_index + 1) * 20.0 + 10.0;
		
    float lweight = 1 - (fabs(lvalue - angle) / 20.0);
    float rweight = 1 - (fabs(rvalue - angle) / 20.0);
		
    atomic_add_local(&(features[index]), lweight * magnitude);
    atomic_add_local(&(features[index + 1]), rweight * magnitude); 
  }

#else
  
  // unweighted histogram calculation 
  int hist_index = floor(angle / 20.0);
  hist_index = (hist_index > 8) ? 8 : hist_index;

  int index = hist_index + cell_index * HISTOGRAM_BINS;

  atomic_add_local(&(features[index]), magnitude);
  
#endif 

  barrier(CLK_LOCAL_MEM_FENCE);

  float norm = 1.0;

#if NORMALIZED

  // block normalization
  int local_index = local_x + local_y * (int)get_local_size(0);
	
  __local float tmp[DESCRIPTOR_SIZE];

  if (local_index < DESCRIPTOR_SIZE) {
    tmp[local_index] = features[local_index] * features[local_index];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
	
  // parallel reduction
  for (int i = DESCRIPTOR_SIZE >> 1; i > 0; i >>= 1) { 
		
    if (local_index < i) { 
      tmp[local_index] += tmp[local_index + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_index == 0) {
    norm = sqrt(tmp[0]);
  }

#endif
	
  // number of windows 
  int total_win_x = (image_width / BLOCK_WIDTH) - (win_width / BLOCK_WIDTH) + 1;
  int total_win_y = (image_height / BLOCK_HEIGHT) - (win_height / BLOCK_HEIGHT) + 1;

  // descriptor writing to global memory on specific index
  if (local_x == 0 && local_y == 0) {		
	
    // calculation of block possible positions
    for (int j = group_y; j > group_y - (win_height / BLOCK_HEIGHT); j--) {
      for (int i = group_x; i > group_x - (win_width / BLOCK_WIDTH); i--) {
				
        if (i < 0 || i >= total_win_x || j < 0 || j >= total_win_y) { 
          continue;
        }
				
        // window index 
        int win_x = i, win_y = j;
        int win_index = win_x + win_y * total_win_x;
        int offset = win_index * ((win_width / BLOCK_WIDTH) * (win_height / BLOCK_HEIGHT) * 36);

        // block index within window
        int block_x = group_x - i;
        int block_y = group_y - j;
        int block_position = DESCRIPTOR_SIZE * (block_x + block_y * (win_width / BLOCK_WIDTH));

        // descriptor position
        int desc_index = offset + block_position;

        for (int n = 0; n < DESCRIPTOR_SIZE; n++) { 
          descriptor[desc_index + n] = features[n] / norm;
        }
      }
    }
  }

#if PREDICTION

  // prediction for window descriptor
  if (local_x == 0 && local_y == 0) {

    for (int j = group_y; j > group_y - (win_height / BLOCK_HEIGHT); j--) {
      for (int i = group_x; i > group_x - (win_width / BLOCK_WIDTH); i--) {
				
        if (i < 0 || i >= total_win_x || j < 0 || j >= total_win_y) { 
          continue;
        }

        // window index
        int win_x = i, win_y = j;
        int win_index = win_x + win_y * total_win_x;
				
        // block index
        int block_x = group_x - i;
        int block_y = group_y - j;
        int block_position = DESCRIPTOR_SIZE * (block_x + block_y * (win_width / BLOCK_WIDTH));

        // product of samples and weights
        for (int n = 0; n < DESCRIPTOR_SIZE; n++) { 
          prediction[win_index] += (features[n] / norm) * y[block_position + n];
        }

        // writing prediction to global memory
        prediction[win_index] += b;
      }
    }
  }
#endif

} 
