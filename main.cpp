//
//  main.cpp
//  mnsit
//
//  Created by 陆闻韬 on 16/4/18.
//  Copyright © 2016年 tobias_lu. All rights reserved.
//

#include <stdio.h>
#include "cnn.hpp"
using namespace std;

//static int reverseInt(int i);
//void loadMnistImage(const char* file, float * data, int num_image);

int main()
{
   // float* data;
   // loadMnistImage("/Users/Tobias_Lu/Documents/data/mnist/train-images.idx3-ubyte", data, 60000);


    cnn c;
    c.initial();
    
    //test kronecker
    
    /*
    float a[9] = {0,1,2,3,4,5,6,7,8};
    float *ret;
    ret = c.Kronecker_2(a, 3);
    for (int i =0;i<3*2;i++)
    {
        for (int j = 0 ; j < 3*2;j++)
        {
            cout<<ret[i * 6 +j]<<" ";
        }
        cout<<endl;
    }
    
    float b[4] = {0,1,2,3};
    float *ret2;
    ret2 = c.Kronecker_2(b, 2);
    for (int i =0;i<2*2;i++)
    {
        for (int j = 0 ; j < 2*2;j++)
        {
            cout<<ret2[i * 4 +j]<<" ";
        }
        cout<<endl;
    }
   */
    c.train();
    
    
    
    return 0;
}

