#include <iostream>
extern "C"

void test_array_increment(float* arr1, const float* const arr2) {
    printf("Arr 2:\n");
    for (int i = 0; i < 3600; i++) {
        arr1[i] = float(i);
        printf("%f, ", arr2[i]);
    }
    printf("\n");
}