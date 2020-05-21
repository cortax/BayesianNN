import sys, os

if __name__ == "__main__":

        with open('array', 'r') as array:
                lines = array.read().splitlines()

        index = int(sys.argv[1])
        time = str(sys.argv[2])
        dataset = str(lines[index])

        os.system('python -m Experiments.HMC --setup='+ dataset + ' --max_time='+time)
