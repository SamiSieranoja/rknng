
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>

typedef int BOOL;
#define true 1
#define false 0


typedef struct kNNItem {
    int id;
    float dist;
    BOOL new_item;
    int visited;
} kNNItem;

typedef struct kNNList {
    // List of <size> number of nearest neighbors
    kNNItem * items;
    float max_dist;
    int size;
    // <id> of point which nearest neighbors this represents
    int id;
    BOOL is_exact;
} kNNList;

typedef struct kNNGraph {
    int size;
    int k;
    int format;
    kNNList * list;
    void * DS;
} kNNGraph ;



extern void printDSVec(kNNGraph* knng,int id);
/*extern kNNGraph* get_knng();*/
/*extern kNNGraph* get_knng(const char* infn,int k, int data_type, int algo, float endcond);*/
//extern kNNGraph* get_knng(const char* infn,int k, int data_type, int algo, float endcond, float nndes_start);
extern kNNGraph* get_knng(const char* infn,int k, int data_type, int algo, float endcond, float nndes_start, int W, int dfunc); 

// extern PyObject* get_knng2(PyArrayObject *aa);

PyObject *__rpdiv_knng(PyArrayObject *py_v, int k, int w, float nndes, float delta, int maxiter); 
PyObject *__rpdiv_knng_o(PyObject *py_v, int k, int w, float nndes, float delta, int maxiter); 

// extern PyObject *__rpdiv_knng(PyArrayObject *py_v, PyObject *py_k, PyObject *py_w, PyObject *py_nndes, PyObject *py_delta, PyObject *py_maxiter); 

extern void test_rknn_lib();
extern float knng_dist(kNNGraph* knng, int p1, int p2);
extern float get_elapsed_time();


/*typedef struct DataSet {*/
/*int size;*/
/*int dimensionality;*/
/*int vector_size;*/
/*float elem_min;*/
/*float elem_max;*/
/*float ** data;*/

/*vector<string>* strings;*/
/*int type;*/
/*} DataSet;*/
