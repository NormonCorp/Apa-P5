from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json


def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        name = parameter["name"]
        print("Capa ", name)
        if name != "classes" and name != "shape_tensor":
            print("procesando ", name)
            s += "parameter:" + str(parameterIndex) + "\n"
            print(parameter["dims"])
            s += "dims:" + str(parameter["dims"]) + "\n"
            print(parameter["name"])
            s += "name:" + str(parameter["name"]) + "\n"
            data = None
            for key in ["doubleData", "floatData", "rawData"]:
                if key in parameter:
                    data = parameter[key]
                    break

            if data is None:
                print("⚠ No se encontró *Data en el parámetro:", parameter.keys())
                data = []  # para no petar

            print(data)
            s += "values:" + str(data) + "\n"

            index = index + 1
            parameterIndex = index // 2
        else:
            print("Esta capa no es interesante ", name)
    return s


def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)


def WriteStandardScaler(file, mean, var):
    """
    Escribe en un fichero de texto las medias y varianzas del StandardScaler
    en el formato que Unity espera:
    - Línea 1: medias separadas por comas
    - Línea 2: varianzas separadas por comas
    """
    with open(file, "w") as f:
        # Primera línea: medias
        f.write(",".join(str(float(m)) for m in mean) + "\n")
        # Segunda línea: varianzas
        f.write(",".join(str(float(v)) for v in var) + "\n")
