#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <pthread.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define LAPLACIAN_THREADS 4     //change the number of threads as you run your concurrency experiment

/* Laplacian filter is 3 by 3 */
#define FILTER_WIDTH 3       
#define FILTER_HEIGHT 3      

#define RGB_COMPONENT_COLOR 255

typedef struct {
      unsigned char r, g, b;
} PPMPixel;

struct parameter {
    PPMPixel *image;         //original image pixel data
    PPMPixel *result;        //filtered image pixel data
    unsigned long int w;     //width of image
    unsigned long int h;     //height of image
    unsigned long int start; //starting point of work
    unsigned long int size;  //equal share of work (almost equal if odd)
};


struct file_name_args {
    char *input_file_name;      //e.g., file1.ppm
    char output_file_name[32];  //will take the form laplaciani.ppm, e.g., laplacian1.ppm
};


/*The total_elapsed_time is the total time taken by all threads 
to compute the edge detection of all input images .
*/
double total_elapsed_time = 0; 
static pthread_mutex_t elapsed_mutex = PTHREAD_MUTEX_INITIALIZER;
static int file_errors = 0;
static pthread_mutex_t error_mutex = PTHREAD_MUTEX_INITIALIZER;


/*This is the thread function. It will compute the new values for the region of image specified in params (start to start+size) using convolution.
    For each pixel in the input image, the filter is conceptually placed on top ofthe image with its origin lying on that pixel.
    The  values  of  each  input  image  pixel  under  the  mask  are  multiplied  by the corresponding filter values.
    Truncate values smaller than zero to zero and larger than 255 to 255.
    The results are summed together to yield a single output value that is placed in the output image at the location of the pixel being processed on the input.
 
 */
void *compute_laplacian_threadfn(void *params)
{
    int laplacian[FILTER_WIDTH][FILTER_HEIGHT] =
    {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };
    struct parameter *p = (struct parameter *)params;

    unsigned long end_row = p->start + p->size;
    unsigned long y, x, ky, kx;

    for (y = p->start; y < end_row; ++y) {
        if (y == 0 || y >= p->h - 1)
            continue;

        for (x = 1; x < p->w - 1; ++x) {
            int red = 0, green = 0, blue = 0;

            for (ky = 0; ky < FILTER_HEIGHT; ++ky) {
                for (kx = 0; kx < FILTER_WIDTH; ++kx) {
                    unsigned long iy = y + ky - 1;
                    unsigned long ix = x + kx - 1;
                    PPMPixel pixel = p->image[iy * p->w + ix];
                    int weight = laplacian[kx][ky];

                    red += pixel.r * weight;
                    green += pixel.g * weight;
                    blue += pixel.b * weight;
                }
            }

            if (red < 0) red = 0;
            if (red > RGB_COMPONENT_COLOR) red = RGB_COMPONENT_COLOR;
            if (green < 0) green = 0;
            if (green > RGB_COMPONENT_COLOR) green = RGB_COMPONENT_COLOR;
            if (blue < 0) blue = 0;
            if (blue > RGB_COMPONENT_COLOR) blue = RGB_COMPONENT_COLOR;

            p->result[y * p->w + x].r = (unsigned char)red;
            p->result[y * p->w + x].g = (unsigned char)green;
            p->result[y * p->w + x].b = (unsigned char)blue;
        }
    }

    return NULL;
}

/* Apply the Laplacian filter to an image using threads.
 Each thread shall do an equal share of the work, i.e. work=height/number of threads. If the size is not even, the last thread shall take the rest of the work.
 Compute the elapsed time and store it in *elapsedTime (Read about gettimeofday).
 Return: result (filtered image)
 */
PPMPixel *apply_filters(PPMPixel *image, unsigned long w, unsigned long h, double *elapsedTime) {
    PPMPixel *result;
    if (w == 0 || h == 0 || w > SIZE_MAX / h ||
        w * h > SIZE_MAX / sizeof(PPMPixel)) {
        fprintf(stderr, "Error: image dimensions are too large\n");
        *elapsedTime = 0.0;
        return NULL;
    }
    result = (PPMPixel *)calloc((size_t)w * h, sizeof(PPMPixel));
    struct timeval start, end;
    pthread_t threads[LAPLACIAN_THREADS];
    struct parameter parameters[LAPLACIAN_THREADS];
    int created_threads[LAPLACIAN_THREADS];
    int thread_count;
    unsigned long row_size, i;

    if (result == NULL) {
        fprintf(stderr, "Error: unable to allocate memory\n");
        *elapsedTime = 0.0;
        return NULL;
    }

    gettimeofday(&start, NULL);

    thread_count = (h == 0) ? 0 :
                   ((h < LAPLACIAN_THREADS) ? (int)h : LAPLACIAN_THREADS);
    if (thread_count == 0) {
        gettimeofday(&start, NULL);
        gettimeofday(&end, NULL);
        *elapsedTime = 0.0;
        return result;
    }

    row_size = h / (unsigned long)thread_count;
    for (i = 0; i < LAPLACIAN_THREADS; ++i)
        created_threads[i] = 0;

    for (i = 0; i < (unsigned long)thread_count; ++i) {
        unsigned long remaining = h - i * row_size;
        unsigned long size = (i == (unsigned long)(thread_count - 1))
                                ? remaining
                                : row_size;

        parameters[i].image = image;
        parameters[i].result = result;
        parameters[i].w = w;
        parameters[i].h = h;
        parameters[i].start = i * row_size;
        parameters[i].size = size;

        if (pthread_create(&threads[i], NULL, compute_laplacian_threadfn,
                           &parameters[i]) == 0) {
            created_threads[i] = 1;
        } else {
            fprintf(stderr, "Error: unable to create Laplacian thread\n");
            compute_laplacian_threadfn(&parameters[i]);
            pthread_mutex_lock(&error_mutex);
            ++file_errors;
            pthread_mutex_unlock(&error_mutex);
        }
    }

    for (i = 0; i < (unsigned long)thread_count; ++i) {
        if (created_threads[i]) {
            int joined = pthread_join(threads[i], NULL);
            if (joined != 0) {
                fprintf(stderr, "Error: unable to join Laplacian thread\n");
                pthread_mutex_lock(&error_mutex);
                ++file_errors;
                pthread_mutex_unlock(&error_mutex);
            }
        }
    }

    gettimeofday(&end, NULL);
    *elapsedTime = (end.tv_sec - start.tv_sec) +
                   (end.tv_usec - start.tv_usec) / 1000000.0;

    return result;
}

/*Create a new P6 file to save the filtered image in. Write the header block
 e.g. P6
      Width Height
      Max color value
 then write the image data.
 The name of the new file shall be "filename" (the second argument).
 */
void write_image(PPMPixel *image, char *filename, unsigned long int width, unsigned long int height)
{
    FILE *fp = fopen(filename, "wb");
    size_t pixel_count;

    if (fp == NULL) {
        fprintf(stderr, "Error: unable to create %s\n", filename);
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
        return;
    }

    if (width == 0 || height == 0 || width > SIZE_MAX / height ||
        width * height > SIZE_MAX / sizeof(PPMPixel)) {
        fprintf(stderr, "Error: invalid image dimensions\n");
        fclose(fp);
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
        return;
    }
    pixel_count = (size_t)width * height;

    if (fprintf(fp, "P6\n%lu %lu\n%d\n", width, height, RGB_COMPONENT_COLOR) < 0) {
        fprintf(stderr, "Error: unable to write %s\n", filename);
        fclose(fp);
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
        return;
    }
    if (fwrite(image, sizeof(PPMPixel), pixel_count, fp) != pixel_count) {
        fprintf(stderr, "Error: unable to write %s\n", filename);
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
        fclose(fp);
        return;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "Error: unable to close %s\n", filename);
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
    }
}



/* Open the filename image for reading, and parse it.
    Example of a ppm header:    //http://netpbm.sourceforge.net/doc/ppm.html
    P6                  -- image format
    # comment           -- comment lines begin with
    ## another comment  -- any number of comment lines
    200 300             -- image width & height 
    255                 -- max color value
 
 Check if the image format is P6. If not, print invalid format error message.
 If there are comments in the file, skip them. You may assume that comments exist only in the header block.
 Read the image size information and store them in width and height.
 Check the rgb component, if not 255, display error message.
 Return: pointer to PPMPixel that has the pixel data of the input image (filename).The pixel data is stored in scanline order from left to right (up to bottom) in 3-byte chunks (r g b values for each pixel) encoded as binary numbers.
 */
PPMPixel *read_image(const char *filename, unsigned long int *width, unsigned long int *height)
{
    FILE *fp;
    char magic[3];
    unsigned int component;
    PPMPixel *img;
    int c;
    unsigned long value;

    #define READ_PPM_TOKEN(out) do {                         \
        do {                                                 \
            c = fgetc(fp);                                   \
            if (c == '#')                                    \
                do { c = fgetc(fp); } while (c != '\n' && c != EOF); \
        } while (c != EOF && isspace((unsigned char)c));    \
        if (c == EOF) goto invalid_header;                   \
        if (ungetc(c, fp) == EOF) goto invalid_header;       \
        if (fscanf(fp, "%lu", &value) != 1) goto invalid_header; \
        *(out) = value;                                      \
    } while (0)

    fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: unable to open %s\n", filename);
        *width = 0;
        *height = 0;
        return NULL;
    }

    if (fscanf(fp, "%2s", magic) != 1 || strcmp(magic, "P6") != 0) {
        fprintf(stderr, "Invalid format\n");
        fclose(fp);
        *width = 0;
        *height = 0;
        return NULL;
    }

    READ_PPM_TOKEN(width);
    READ_PPM_TOKEN(height);
    READ_PPM_TOKEN(&component);
    if (component != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "Invalid RGB component\n");
        fclose(fp);
        *width = 0;
        *height = 0;
        return NULL;
    }

    #undef READ_PPM_TOKEN

    c = fgetc(fp);
    if (c == '\r') {
        int next = fgetc(fp);
        if (next != '\n')
            ungetc(next, fp);
    } else if (c == EOF) {
        fclose(fp);
        *width = 0;
        *height = 0;
        return NULL;
    }

    if (*width == 0 || *height == 0)
        goto invalid_header;

    if (*width > SIZE_MAX / *height ||
        (*width * *height) > SIZE_MAX / sizeof(PPMPixel)) {
        fprintf(stderr, "Error: image dimensions are too large\n");
        fclose(fp);
        *width = 0;
        *height = 0;
        return NULL;
    }

    img = (PPMPixel *)malloc((size_t)(*width) * (*height) * sizeof(PPMPixel));
    if (img == NULL) {
        fprintf(stderr, "Error: unable to allocate memory\n");
        fclose(fp);
        return NULL;
    }

    if (fread(img, sizeof(PPMPixel), (size_t)(*width) * (*height), fp)
        != (size_t)(*width) * (*height)) {
        fprintf(stderr, "Error: incomplete image data\n");
        free(img);
        fclose(fp);
        *width = 0;
        *height = 0;
        return NULL;
    }

    fclose(fp);
    return img;

invalid_header:
    fprintf(stderr, "Invalid format\n");
    fclose(fp);
    *width = 0;
    *height = 0;
    return NULL;
}

/* The thread function that manages an image file. 
 Read an image file that is passed as an argument at runtime. 
 Apply the Laplacian filter. 
 Update the value of total_elapsed_time.
 Save the result image in a file called laplaciani.ppm, where i is the image file order in the passed arguments.
 Example: the result image of the file passed third during the input shall be called "laplacian3.ppm".

*/
void *manage_image_file(void *args){
    struct file_name_args *file_args = (struct file_name_args *)args;
    unsigned long width, height;
    PPMPixel *image;
    PPMPixel *result;
    double elapsed;

    image = read_image(file_args->input_file_name, &width, &height);
    if (image == NULL) {
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
        return NULL;
    }

    result = apply_filters(image, width, height, &elapsed);
    free(image);

    if (result != NULL) {
        write_image(result, file_args->output_file_name, width, height);
        free(result);
    } else {
        pthread_mutex_lock(&error_mutex);
        ++file_errors;
        pthread_mutex_unlock(&error_mutex);
    }

    pthread_mutex_lock(&elapsed_mutex);
    total_elapsed_time += elapsed;
    pthread_mutex_unlock(&elapsed_mutex);
    return NULL;
}
/*The driver of the program. Check for the correct number of arguments. If wrong print the message: "Usage ./edge_detector filename[s]"
  It shall accept n filenames as arguments, separated by whitespace, e.g., ./edge_detector file1.ppm file2.ppm    file3.ppm
  It will create a thread for each input file to manage.  
  It will print the total elapsed time in .4 precision seconds(e.g., 0.1234 s). 
 */
int main(int argc, char *argv[])
{
    pthread_t *threads;
    struct file_name_args *file_args;
    int *started;
    int i;

    if (argc < 2) {
        printf("Usage: ./edge_detector filename[s]\n");
        return 1;
    }

    if ((size_t)(argc - 1) > SIZE_MAX / sizeof(pthread_t) ||
        (size_t)(argc - 1) > SIZE_MAX / sizeof(struct file_name_args) ||
        (size_t)(argc - 1) > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "Error: too many input files\n");
        return 1;
    }

    threads = (pthread_t *)malloc((size_t)(argc - 1) * sizeof(pthread_t));
    file_args = (struct file_name_args *)malloc((size_t)(argc - 1)
                                                * sizeof(struct file_name_args));
    started = (int *)calloc((size_t)(argc - 1), sizeof(int));
    if (threads == NULL || file_args == NULL || started == NULL) {
        free(threads);
        free(file_args);
        free(started);
        fprintf(stderr, "Error: unable to allocate memory\n");
        return 1;
    }

    for (i = 0; i < argc - 1; ++i) {
        file_args[i].input_file_name = argv[i + 1];
        snprintf(file_args[i].output_file_name,
                 sizeof(file_args[i].output_file_name),
                 "laplacian%d.ppm", i + 1);

        if (pthread_create(&threads[i], NULL, manage_image_file,
                           &file_args[i]) != 0) {
            fprintf(stderr, "Error: unable to create thread\n");
            pthread_mutex_lock(&error_mutex);
            ++file_errors;
            pthread_mutex_unlock(&error_mutex);
        } else {
            started[i] = 1;
        }
    }

    for (i = 0; i < argc - 1; ++i) {
        if (started[i]) {
            if (pthread_join(threads[i], NULL) != 0) {
                fprintf(stderr, "Error: unable to join image thread\n");
                pthread_mutex_lock(&error_mutex);
                ++file_errors;
                pthread_mutex_unlock(&error_mutex);
            }
        }
    }

    printf("%.4f s\n", total_elapsed_time);

    free(file_args);
    free(threads);
    free(started);
    return file_errors == 0 ? 0 : 1;
}

