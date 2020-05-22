import sys, os

if __name__ == "__main__":

        with open('array', 'r') as array:
                lines = array.read().splitlines()

        index = int(sys.argv[1])
        time = str(sys.argv[2])
        line = lines[index+1].split(',')
        dataset = str(line[0])
        num_iter = str(line[1])
        burning = str(line[2])
        thinning = str(line[3])

        os.system('python -m Experiments.HMC --setup='+ dataset + ' --num_iter=' + num_iter + ' --burning=' + burning + ' --thinning=' + thinning + ' --max_time='+ time)
