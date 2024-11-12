# imports
import streamlit as st
import ast
import base64
import io
from io import BytesIO
import requests
import urllib
import numpy as np
import random

from clarifai.client.app import App
from clarifai.client.auth import create_stub
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.input import Inputs
from clarifai.client.model import Model
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from annotated_text import annotated_text

from concurrent.futures import ThreadPoolExecutor, as_completed
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from PIL import Image, ImageDraw, ImageFont, ImageOps

from streamlit_image_select import image_select

# streamlit config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

PAT = st.secrets["CLARIFAI_PAT"]
USER_ID = 'clarifai'

# setup
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
metadata = (('authorization', 'Key ' + PAT),)

##########################
#### HELPER FUNCTIONS ####
##########################

def text_color_for_background(hex_code):
    """Determine the appropriate text color (white or black) for a given background color."""
    return "#000000" if is_light_or_dark(hex_code) == "light" else "#ffffff"

def footer(st):
  with open('footer.html', 'r') as file:
    footer = file.read()
    st.write(footer, unsafe_allow_html=True)

def url_picture_to_base64(img, pat):
    headers = {'Authentication': f'Key {pat}'}
    response = requests.get(img, headers=headers)
    img_byte = BytesIO(response.content).read()
    return img_byte

def post_model_output(stub, user_app_id, model_id, version_id, inputs, auth_metadata):
    print(f"user_app_id:{user_app_id}")
    print(f"model_id:{model_id} version_id: {version_id}")

    res_pmo = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=user_app_id,
            model_id=model_id,
            version_id=version_id,
            inputs=inputs
        ),
        metadata=auth_metadata
    )
    return res_pmo

def fix_base64_padding(base64_string):
    """Fix base64 string padding by adding '=' characters if needed."""
    missing_padding = len(base64_string) % 4
    if missing_padding:
        base64_string += '=' * (4 - missing_padding)
    return base64_string

def display_segmented_image(pred_response, SEGMENT_IMAGE_URL):
    """Displays the segmented part of the image using the model response."""
    try:
        # Load original image using PIL
        response = requests.get(SEGMENT_IMAGE_URL)
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')  # Ensure image is in RGB format
        img_array = np.array(img)

        # Extract regions
        regions = pred_response.outputs[0].data.regions
        masks = []
        concepts = []

        # Check if any regions were detected
        if not regions:
            st.text("No cracks found")
            return

        for region in regions:
            masks.append(np.array(Image.open(BytesIO(region.region_info.mask.image.base64))))
            concepts.append(region.data.concepts[0])  # Store the whole concept object

        # Create overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Combine all masks
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask > 0)

        # Convert combined mask to image
        mask_image = Image.fromarray((combined_mask * 255).astype('uint8'))
        
        # Create green overlay
        green_overlay = Image.new('RGBA', img.size, (0, 255, 0, 102))  # 102 is 0.4 opacity in 255 scale
        
        # Apply mask to green overlay
        final_overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        final_overlay.paste(green_overlay, mask=mask_image)

        # Composite the images
        result = Image.alpha_composite(img.convert('RGBA'), final_overlay)
        result = result.convert('RGB')  # Convert back to RGB for display

        # Display the result using Streamlit
        st.image(result, use_column_width=True)

        # Display confidence scores using annotated_text
        annotation_data = []
        tag_bg_color_1 = "#00815f"  # You can adjust these colors
        tag_text_color_1 = "#ffffff"

        # Create annotation data
        for concept in concepts:
            annotation_data.append(
                (f'{concept.name}', f'{concept.value:.3f}', tag_bg_color_1, tag_text_color_1)
            )

        # Create list with spaces between annotations
        list_with_empty_strings = []
        for item in annotation_data:
            list_with_empty_strings.append(item)
            list_with_empty_strings.append(" ")

        # Remove trailing space if it exists
        if list_with_empty_strings and list_with_empty_strings[-1] == " ":
            list_with_empty_strings.pop()

        # Display annotations
        st.write("Confidence Scores:")
        annotated_text(*tuple(list_with_empty_strings))

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

####################
####  SIDEBAR   ####
####################

with st.sidebar:
    st.caption('Below options are mostly here to help customize any displayed graphics/text.')

    with st.expander('Header Setup'):
        company_logo = st.text_input(label='Banner Url', value='https://upload.wikimedia.org/wikipedia/commons/b/bc/Clarifai_Logo_FC_Web.png')
        company_logo_width = st.slider(label='Banner Width', min_value=1, max_value=1000, value=300)
        page_title = st.text_input(label='Module Title', value='Visual Inspection Demo')

    with st.expander('Anamoly Detection'):
        anamoly_detection_subheader_title = st.text_input(label='Anamoly Detection subheader title', value='✨ Leveraging Clarifai for Anamoly Detection ✨')
        anamoly_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/d364a803a427d30363defabb5ac9cdb6?t=1683122597793\nhttps://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/aafd51c3511a17aad284987942cdb5c2?t=1683122612382\nhttps://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/bce40a06cb39ecd513d277ecdf3ab0a7?t=1683128606787\nhttps://s3.amazonaws.com/clarifai-api/img3/prod/large/c47394b31b564cf49f0157fc87ff7c3a/f3f74ebfec4d67b3959cde25575161a0?t=1683122628342'
        )

    with st.expander('Insulator Defect Detection'):
        defect_detection_subheader_title = st.text_input(label='Insulator Defect Detection Subheader Text', value='✨ Leveraging Clarifai for Insulator Defect Detection ✨')
        defect_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_1.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_2.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_3.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/defect_detection_4.jpeg'
        )

        threshold = st.slider(label='Defect Threshold', min_value=0.0, max_value=1.0, value=0.3)
        tag_bg_color_1 = st.color_picker(label='Tag Background Color', value='#aabbcc', key='tag_bg_color_1')
        tag_text_color_1 = st.color_picker(label='Tag Text Color', value='#2B2D37', key='tag_text_color_1')

    with st.expander('Crack Segmentation'):
        segmentation_subheader_title = st.text_input(label='Crack Segmentation Subheader Text', value='✨ Leveraging Clarifai to segment cracked parts ✨')
        crack_images = st.text_area(height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_1.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_2.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_3.jpeg\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/crack_4.jpeg'
        )

    with st.expander('Surface Defect Detection'):
        surface_defect_detection_subheader_title = st.text_input(label='Surface Defect Detection Subheader Text', value='✨ Leveraging Clarifai for Metal Surface Defect Detection ✨')
        surface_images = st.text_area(
            height = 300,
            label = 'Prepopulated Carousel Images.',
            help = "One URL per line. No quotations. Underlying code will take in the entire text box's value as a single string, then split using `theTextString.split('\n')`",
            value = 'https://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_1.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_2.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_3.png\nhttps://s3.us-east-1.amazonaws.com/samples.clarifai.com/surface_4.png',
          )

        st.subheader('Output Display Options')
        tag_bg_color_2 = st.color_picker(label='Tag Background Color', value='#aabbcc', key='tag_bg_color_2')
        tag_text_color_2 = st.color_picker(label='Tag Text Color', value='#2B2D37', key='tag_text_color_2')

####################
####  MAIN PAGE ####
####################

st.image(company_logo, width=company_logo_width)
st.title(page_title)

tab1, tab2, tab3, tab4 = st.tabs(['Anamoly Detection', 'Insulator Defect Detection', 'Crack Segmentation', 'Surface Defect Detection'])

##############################
#### Anamoly Detection ####
##############################

with tab1:
    try:
        st.subheader(anamoly_detection_subheader_title)
        
        # Select example image
        img = image_select(
            label="Select an image:",
            images=anamoly_images.split('\n'),
            captions=["Chipped #1", "Chipped #2", "Chipped #3", "Dirty"]
        )

        if st.button("Run Anomaly Detection"):
            st.divider()
            
            app_id = 'anomaly-detection-tablet-pills'
            model_id = 'pill-anomaly'
            version_id = '38caead067764267bcfc1b0974cf3488'
            
            userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=app_id)
            
            # Create input proto
            inp = resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        base64=url_picture_to_base64(img, auth._pat)
                    )
                )
            )
            
            with st.spinner("Processing anomaly detection..."):
                res_pmo = post_model_output(stub, userDataObject, model_id, version_id, [inp], metadata)
                output_heatmap = res_pmo.outputs[0].data.heatmaps[0].base64
                heatmap_im = Image.open(BytesIO(output_heatmap))

                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)

                with col2:
                    st.write('Heatmap')
                    heatmap_im_color = ImageOps.colorize(heatmap_im, black='red', white='black')
                    heatmap_im_color = heatmap_im_color.resize(im1_pil.size, resample=0)
                    st.image(heatmap_im_color)

                with col3:
                    st.write("Composite")
                    mask = Image.new("L", im1_pil.size, 64)
                    composite_im = Image.composite(im1_pil, heatmap_im_color, mask)
                    st.image(composite_im)
    
    except Exception as e:
        st.error(f"Error in Anomaly Detection tab: {str(e)}")

#########################
#### Defect Detection ####
#########################

with tab2:
    try:
        st.subheader(defect_detection_subheader_title)
        
        img = image_select(
            label="Select image:",
            images=defect_images.split('\n'),
            captions=["#1", "#2", "#3", "#4"]
        )

        if st.button("Run Defect Detection"):
            st.divider()
            
            app_id = "insulator-defect-detection"
            model_id = "insulator-condition-inception"
            version_id = "810df853edb942d3ae45399746479ab6"
            
            userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=app_id)
            
            inp = resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        base64=url_picture_to_base64(img, auth._pat)
                    )
                )
            )
            
            with st.spinner("Processing defect detection..."):
                res_pmo = post_model_output(stub, userDataObject, model_id, version_id, [inp], metadata)
                outputs = res_pmo.outputs[0]
                regions = outputs.data.regions

                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)

                with col2:
                    st.write("Predicted Defects")
                    image = Image.open(urllib.request.urlopen(img))
                    width, height = image.size
                    line_passes = 3
                    
                    threshold = threshold

                    concept_data = []

                    annotation_data = []

                    for region in regions:
                        top_row = round(region.region_info.bounding_box.top_row, 3)
                        left_col = round(region.region_info.bounding_box.left_col, 3)
                        bottom_row = round(region.region_info.bounding_box.bottom_row, 3)
                        right_col = round(region.region_info.bounding_box.right_col, 3)

                        for concept in region.data.concepts:
                            if concept.value >= threshold:
                              concept_data.append({
                                      "type": concept.name, 
                                      "confidence": concept.value,
                                      "width": width,
                                      "height": height,
                                      "top_row": top_row,
                                      "left_col": left_col,
                                      "bottom_row": bottom_row,
                                      "right_col": right_col
                                    })
                              annotation_data.append(
                                (f'{concept.name}', f'{concept.value:.3f}', tag_bg_color_1, tag_text_color_1)
                              )
                        img1 = ImageDraw.Draw(image)

                        for concept in concept_data:
                          # create rectangle image
                          h1 = concept["top_row"]    * height
                          w1 = concept["left_col"]   * width
                          h2 = concept["bottom_row"] * height
                          w2 = concept["right_col"]  * width

                          img1.rectangle((w1, h1, w2, h2), width=line_passes, outline='blue')

                          # calculate font size based off of image height
                          fontsize = int(image.height / 40)
                          font = ImageFont.load_default()

                          offsets = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]  # 8 directions
                          for offset_x, offset_y in offsets:
                                img1.text(
                                    (w1 + offset_x + line_passes, h1 + offset_y + line_passes),
                                    concept["type"],
                                    align="left",
                                    fill='black',  # Border color
                                    font=font
                                )
                          img1.text(
                                    (w1 + line_passes, h1 + line_passes),
                                    concept["type"],
                                    align="left",
                                    fill='white',  # Text fill color
                                    font=font
                                )

                    st.image(image)
                    # Add spaces between annotations
                    list_with_empty_strings = []
                    for item in annotation_data:
                        list_with_empty_strings.append(item)
                        list_with_empty_strings.append(" ")
                    
                    # Remove trailing space if it exists
                    if list_with_empty_strings and list_with_empty_strings[-1] == " ":
                        list_with_empty_strings.pop()
                    
                    # Display annotations
                    st.write("Detected Regions and Confidence Scores:")
                    annotated_text(*tuple(list_with_empty_strings))

    except Exception as e:
        st.error(f"Error in Defect Detection tab: {str(e)}")

######################
#### Crack Segmentation ####
######################

with tab3:
    try:
        st.subheader(segmentation_subheader_title)
        
        img = image_select(
            label="Select the image:",
            images=crack_images.split('\n'),
            captions=["Crack #1", "Crack #2", "Crack #3", "Crack #4"]
        )

        if st.button("Run Crack Segmentation"):
            st.divider()
            
            app_id = 'crack-segmentation'
            model_id = 'crack-segmentation'
            version_id = 'f5efe6a57e7e4a3d922635d5523a2a7a'
            
            userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=app_id)
            
            inp = resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        base64=url_picture_to_base64(img, auth._pat)
                    )
                )
            )
            
            with st.spinner("Processing crack segmentation..."):
                res_pmo = post_model_output(stub, userDataObject, model_id, version_id, [inp], metadata)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)

                with col2:
                    st.write('Segmented Image')
                    display_segmented_image(res_pmo, img)
    
    except Exception as e:
        st.error(f"Error in Crack Segmentation tab: {str(e)}")

#########################
#### Surface Defect Detection ####
#########################

with tab4:
    try:
        st.subheader(surface_defect_detection_subheader_title)
        
        img = image_select(
            label="Select image:",
            images=surface_images.split('\n'),
            captions=["Surface #1", "Surface #2", "Surface #3", "Surface #4"]
        )

        if st.button("Run Surface Defect Detection"):
            st.divider()
            
            app_id = "surface-defects-sheet-metal"
            model_id = "surface-defects"
            version_id = "edde66fb372548ad85cd830af0c55477"
            
            userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=app_id)
            
            inp = resources_pb2.Input(
                data=resources_pb2.Data(
                    image=resources_pb2.Image(
                        base64=url_picture_to_base64(img, auth._pat)
                    )
                )
            )
            
            with st.spinner("Processing surface defect detection..."):
                surface_class_pred = post_model_output(stub, userDataObject, model_id, version_id, [inp], metadata)

                col1, col2 = st.columns(2)
                
                with col1:
                    st.write('Original')
                    im1_pil = Image.open(urllib.request.urlopen(img))
                    st.image(im1_pil)

                with col2:
                    st.write('Surface Defect Detection Results')
                    
                    # Create tuples for annotation
                    concept_data = tuple([
                        (f'{x.name}', f'{x.value:.3f}', tag_bg_color_2, tag_text_color_2) 
                        for x in surface_class_pred.outputs[0].data.concepts
                    ])

                    # Add spaces between annotations
                    list_with_empty_strings = []
                    for item in concept_data:
                        list_with_empty_strings.append(item)
                        list_with_empty_strings.append(" ")
                    
                    if list_with_empty_strings and list_with_empty_strings[-1] == " ":
                        list_with_empty_strings.pop()
                    
                    concept_data = tuple(list_with_empty_strings)
                    annotated_text(*concept_data)
    
    except Exception as e:
        st.error(f"Error in Surface Defect Detection tab: {str(e)}")

####################
####  FOOTER    ####
####################

footer(st)
