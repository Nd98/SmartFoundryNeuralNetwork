import BackPropagation as bp

def train(output_para,input_para,f,neurons,tf):
    x = output_para
    while(x>0):
        (bp.BackProp(x,output_para,input_para,f,100,int(neurons),tf))
        x-=1
    return "Model Trained"

def prediction(arr,output_para,f,no_of_output_para):
    result = []
    x = output_para
    while(x>0):
        result.append(bp.predict(arr,x,f,no_of_output_para))
        x-=1
    print(result)
    return result[::-1]