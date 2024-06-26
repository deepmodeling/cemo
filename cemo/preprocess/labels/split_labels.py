import pickle

def split_labels(input_file_name:str, output_file_names:str,n_types:int,start_num:int=0):
    '''
    Split labels into index files.
    '''
    fin=pickle.load(open(input_file_name,'rb'))
    #ans=[]
    #for _ in range(n_types):
    #    ans.append([]) # can't be instead of map because [] would be treat as same object
    ans=list(map(lambda _: [], range(n_types)))
    func_proc=lambda x: ans[fin[x]-start_num].append(x)
    n_labels=len(fin)
    list(map(func_proc,range(n_labels)))
    func_save=lambda x:pickle.dump(ans[x],open(output_file_names%x,'wb'))
    list(map(func_save,range(n_types)))