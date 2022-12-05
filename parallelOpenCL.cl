typedef struct{
	float x;
	float y;
} floatvector;

// Stores rendered colors.
typedef struct{
	float red;
	float green;
	float blue;
} color;

// Stores the satellite data, which fly around black hole in the space
typedef struct{
	color identifier;
	floatvector position;
	floatvector velocity;
} satellite;
 
#define WINDOW_WIDTH  1024
#define WINDOW_HEIGHT 1024
#define SATELLITE_COUNT 64
#define SATELLITE_RADIUS 3.16f
#define SIZE WINDOW_HEIGHT * WINDOW_HEIGHT



__kernel void parallelOpenCL(__global satellite *satellites, __global color* pixelsOut) {
	

	int idx = get_global_id(0);
	int idy = get_global_id(1);
	
		// Row wise ordering
		floatvector pixel = {.x = idx, .y = idy};

		// This color is used for coloring the pixel
		color renderColor = {.red = 0.f, .green = 0.f, .blue = 0.f};

		// Find closest satellite
		float shortestDistance = INFINITY;

		float weights = 0.f;
		int hitsSatellite = 0;
      
		// First Graphics satellite loop: Find the closest satellite.
		for(int j = 0; j < SATELLITE_COUNT; ++j) {
			floatvector difference = {.x = pixel.x - satellites[j].position.x,
									.y = pixel.y - satellites[j].position.y};
			float distance = sqrt(difference.x * difference.x + 
								difference.y * difference.y);

			if(distance < SATELLITE_RADIUS) {
			renderColor.red = 1.0f;
			renderColor.green = 1.0f;
			renderColor.blue = 1.0f;
			hitsSatellite = 1;
			break;
			} else {
				float weight = 1.0f / (distance*distance*distance*distance);
				weights += weight;
				if(distance < shortestDistance) {
					shortestDistance = distance;
					renderColor = satellites[j].identifier;
				}
			}
		}

		// Second graphics loop: Calculate the color based on distance to every satellite.
		if (!hitsSatellite) {
			for(int j = 0; j < SATELLITE_COUNT; ++j){

				floatvector difference = {.x = pixel.x - satellites[j].position.x,
											.y = pixel.y - satellites[j].position.y};
				float dist2 = (difference.x * difference.x +
								difference.y * difference.y);
				float weight = 1.0f/(dist2* dist2);

				renderColor.red += (satellites[j].identifier.red *
									weight /weights) * 3.0f;

				renderColor.green += (satellites[j].identifier.green *
										weight / weights) * 3.0f;

				renderColor.blue += (satellites[j].identifier.blue *
										weight / weights) * 3.0f;
			}
		}
		
		
		pixelsOut[idx + WINDOW_WIDTH * idy] = renderColor;
}