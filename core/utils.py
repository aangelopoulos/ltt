import os
import pathlib
import pickle as pkl

def cacheable(func):
    def cache_func(*args):
        fname = str(pathlib.Path(__file__).parent.absolute()) + '/.cache/' + str(func).split(' ')[1] + str(args) + '.pkl'
        os.makedirs('./.cache/', exist_ok=True)
        if os.path.exists(fname):
            filehandler = open(fname, 'rb')
            return pkl.load(filehandler)    
        else:
            filehandler = open(fname, 'wb')
            result = func(*args)
            pkl.dump(result, filehandler)   
            return result

    return cache_func

if __name__=="__main__":
    print("Hello, World!")
