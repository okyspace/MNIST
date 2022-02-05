# to extend with more config possibilities
def create_config_pbtxt_tensorflow(model, config_pbtxt_file):
    platform = "tensorflow_savedmodel"
    input_name = model.input_names[0]
    output_name = model.output_names[0]
    input_data_type = "TYPE_FP32"
    output_data_type = "TYPE_FP32"
    input_dims = str(model.input.shape.as_list()).replace("None", "-1")
    output_dims = str(model.output.shape.as_list()).replace("None", "-1")

    config_pbtxt = """
  
        platform: "%s"
        input [
            {
                name: "%s"
                data_type: %s
                dims: %s
            }
        ]
        output [
            {
                name: "%s"
                data_type: %s
                dims: %s
            }
        ]
    """ % (
      
        platform,
        input_name, input_data_type, input_dims,
        output_name, output_data_type, output_dims
    )

    with open(config_pbtxt_file, "w") as config_file:
        config_file.write(config_pbtxt)


# add in input format for sample.
def convert2torchscript(sample, model):
    # TODO


# to extend with more config possibilities (dyanmic batching, model instances, versions) and variable inputs/outputs
# incomplete; todo
def create_config_pbtxt_torchscript(model, config_pbtxt_file):
    platform = "pytorch_libtorch"
    input_name = 'input__0'
    output_name = 'output__0'
