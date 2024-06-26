import pickle
from functools import reduce

def concat_labels(input_file_names:str, output_file_name:str,nums:list):    
    data=list(map(lambda x:pickle.load(open(x,'rb')),[input_file_names%i for i in nums]))
    #result=[]
    #map(lambda x:result.extend(x),data[1:])
    result=list(reduce(lambda x,y: x+y,data))
    result.sort()
    pickle.dump(result,open(output_file_name,'wb'))