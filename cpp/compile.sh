TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -shared high_dim_filter.cc modified_permutohedral.cc -o high_dim_filter.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2