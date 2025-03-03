import os

INTERPRETER = "/home/vin/yes/envs/tf_gpu/bin/python"


def main():
    for i in range(1):
        #print('copied')
        #copyfile('populations/populations/metis_dataset' + str(i) + '.h5', 'original_dataset/metis_dataset.h5')
        #os.system(INTERPRETER+' gen_metis.py [1] 0.75 10 metis-experiment'+str(i)+' 3')
        #os.system(INTERPRETER + ' dist_gen_diff.py occl 3 .5 0 .1 50 20 .25 --target_model=0')
        #os.system(INTERPRETER + ' dist_gen_diff.py light 1 .1 0 .1 50 10 0 --target_model=0')
        #os.system(INTERPRETER + ' dist_gen_diff.py occl 1 .1 10 50 20 0 --target_model=0')

        os.system(INTERPRETER + ' dist_gen_diff.py light 1 .1 0 .1 50 2500 0 --target_model=0')


if __name__ == "__main__":
    main()
