#include <iostream>
#include <omp.h> // OpenMPのヘッダをinclude

int main(){
//ここが並列処理される
  #pragma omp parallel
  {
    std::cout << "Hello World!\n";
  }
}