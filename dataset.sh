dataset_name="monet2photo"

cd ../datasets && \
    curl -O "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/${dataset_name}.zip" && \
    unzip {dataset_name}.zip && \
    rm {dataset_name}.zip && \
    mkdir train && \
    mkdir test && \
    mv ./trainA ./train/trainA && \
    mv ./trainB ./train/trainB && \
    mv ./testA ./test/testA && \
    mv ./testB ./test/testB