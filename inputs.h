struct Params{
    int batch, dim, width, height;

    void *__restrict__ in_ptr;
    void *__restrict__ out_ptr;
};
