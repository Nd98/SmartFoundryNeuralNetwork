import BackPropagation as bp

def train(output_para,input_para,f,neurons,tf,ta):

    if ta == "bp":
        x = output_para
        while(x>0):
            (bp.BackProp(x,output_para,input_para,f,100,int(neurons),tf))
            x-=1
        return "Model Trained"
    
    elif ta == "qn":
        x = output_para
        while(x>0):
            (bp.QuasiNewton(x,output_para,input_para,f,100,int(neurons),tf))
            x-=1
        return "Model Trained"
    
    elif ta == "lm":
        x = output_para
        while(x>0):
            (bp.LevenbergMarquardt(x,output_para,input_para,f,100,int(neurons),tf))
            x-=1
        return "Model Trained"

    elif ta == "ma":
        x = output_para
        while(x>0):
            (bp.MomentumAdaptation(x,output_para,input_para,f,100,int(neurons),tf))
            x-=1
        return "Model Trained"

def prediction(arr,output_para,f,no_of_output_para):
    result = []
    x = output_para
    while(x>0):
        r = bp.predict(arr,x,f,no_of_output_para)
        if r != None:
            result.append(r)
        x-=1
    print(result)
    return result[::-1]