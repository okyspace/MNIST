"""This module contains utility codes for interacting with Triton server"""
# TODO: To extend with more config possibilities
def create_config_pbtxt_tensorflow(model, config_pbtxt_file):
    """
    DOCSTRING TO BE INCLUDED!
    """

    platform = "tensorflow_savedmodel"
    input_name = model.input_names[0]
    output_name = model.output_names[0]
    input_data_type = "TYPE_FP32"
    output_data_type = "TYPE_FP32"
    input_dims = str(model.input.shape.as_list()).replace("None", "-1")
    output_dims = str(model.output.shape.as_list()).replace("None", "-1")

    # String is meant to look like a file and is simpler to write using literals.
    # pylint: disable-next=consider-using-f-string
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
        input_name,
        input_data_type,
        input_dims,
        output_name,
        output_data_type,
        output_dims,
    )

    with open(config_pbtxt_file, "w", encoding="utf-8") as config_file:
        config_file.write(config_pbtxt)


# TODO: Add in input format for sample.
def convert2torchscript():
    """
    DOCSTRING TO BE INCLUDED!
    Parameters (Not included in function yet due to non reference)
    ----------
    sample:
    model:
    """


# TODO: To extend with more config possibilities (dyanmic batching, model instances, versions)
# and variable inputs/outputs
def create_config_pbtxt_torchscript():
    """
    DOCSTRING TO BE INCLUDED!
    Parameters (Not included in function yet due to non reference)
    ----------
    model:
    config_pbtxt_file:
    """
    # platform = "pytorch_libtorch"
    # input_name = "input__0"
    # output_name = "output__0"
