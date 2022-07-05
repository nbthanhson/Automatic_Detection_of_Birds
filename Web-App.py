import os
import torch
import cv2
import json
import shutil
from src import titleEN
from src import titleVN
import numpy as np
import streamlit as st
from PIL import Image

@st.cache()
def load_model(path: str = 'weights/best.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = path)
    return model

@st.cache()
def load_file_structure(path: str = 'src/all_imgs.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)

@st.cache()
def load_list_of_images(
        all_images: dict,
        image_files_dtype: str,
        bird_species: str
        ) -> list:
    species_dict = all_images.get(image_files_dtype)
    list_of_files = species_dict.get(bird_species)
    return list_of_files

@st.cache(allow_output_mutation = True)
def get_prediction(img_bytes, model):
    results = model(img_bytes)  
    return results

def main():
    st.set_page_config(
        page_title = "Automatic Detection of Birds",
        page_icon = "🔎",
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    model = load_model()
    all_images = load_file_structure()
    types_of_birds = sorted(list(all_images['train'].keys()))
   
    dtype_file_structure_mapping = {
        'All Images': 'all_image',
        'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid',
        'Images The Model Has Never Seen': 'test'
    }
    data_split_names = list(dtype_file_structure_mapping.keys())

    with st.sidebar:
        st.image(Image.open('imgs/Logo02.png'), width = 100)

        select_page = st.radio("SELECT PAGE", ["HOME PAGE", "ABOUT ADB", "CONTACT"])
        st.markdown("<br /><br /><br /><br /><br /><br />", unsafe_allow_html = True)
        st.markdown("<hr />", unsafe_allow_html = True)
        language = st.selectbox("LANGUAGE", ["English", "Vietnamese"])

    if language == "English":
        if select_page == "HOME PAGE":
            col1, col2 = st.columns([8.1, 4])
            file_img, file_vid, key_path = '', '', ''

            with col2:
                logo = Image.open('imgs/Logo01.png')
                st.image(logo, use_column_width = True)

                with st.expander("How to use ADB?", expanded = True):
                    st.markdown(titleEN.STORY, unsafe_allow_html = True)
    
            with col1:
                st.markdown(titleEN.CHEF_INFO, unsafe_allow_html=True)

                choice_way = st.radio("Pick one", [
                                    "Upload an image",
                                    "Choose from available images"])

                if choice_way == "Upload an image":
                    file_img = st.file_uploader("Upload an Image of Birds")

                    if file_img:
                        img = Image.open(file_img)

                else:
                    dataset_type = st.selectbox("Data Portion Type", data_split_names)
                    data_folder = dtype_file_structure_mapping[dataset_type]

                    selected_species = st.selectbox("Bird Type", types_of_birds)
                    available_images = load_list_of_images(all_images, data_folder, selected_species)
                    image_name = st.selectbox("Image Name", available_images)

                    key_path = os.path.join('bird_dataset', data_folder, image_name)
                    img = cv2.imread(key_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                recipe_button = st.button('Get Result!')

            st.markdown("<hr />", unsafe_allow_html = True)

            if recipe_button:
                with st.spinner("Wait for it..."):
                    if file_img or key_path:
                        col3, col4 = st.columns([5, 4])
                        with col3:
                            if os.path.isdir('./runs'):
                                shutil.rmtree('./runs')

                            results = get_prediction(img, model)
                            results.save()

                            st.header("Here is the result!")

                            img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                            img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                            st.image(img_res, use_column_width = True)

                            df = results.pandas().xyxy[0]
                            del df['class']
                            st.write(df)
                        
                        with col4:
                            st.header("Description")

                            des = set()
                            for name_type in df['name']:
                                if name_type not in des:
                                    if name_type == 'bird':
                                        continue

                                    elif name_type == 'eagle':
                                        with st.expander("Eagle"):
                                            st.markdown(titleEN.EAGLE, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)
                                    
                                    elif name_type == 'flamingo':
                                        with st.expander("Flamingo"):
                                            st.markdown(titleEN.FLAMINGO, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    elif name_type == 'golden oriole':
                                        with st.expander("Golden Oriole"):
                                            st.markdown(titleEN.GOLDEN_ORIOLE, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    elif name_type == 'hyacinth macaw':
                                        with st.expander("Hyacinth Macaw"):
                                            st.markdown(titleEN.HYACINTH_MACAW, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)
                                    
                                    elif name_type == 'mute swan':
                                        with st.expander("Mute Swan"):
                                            st.markdown(titleEN.MUTE_SWAN, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    elif name_type == 'seagull':
                                        with st.expander("Seagull"):
                                            st.markdown(titleEN.SEAGULL, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    des.add(name_type)

                            if not des:
                                st.info("There is no data to describe!")

                    else:
                        st.error('No image available. Please choose an image of bird!')
        
    else:
        if select_page == "HOME PAGE":
            col1, col2 = st.columns([8.1, 4])
            file_img, file_vid, key_path = '', '', ''

            with col2:
                logo = Image.open('imgs/Logo01.png')
                st.image(logo, use_column_width = True)

                with st.expander("Cách sử dụng ADB?", expanded = True):
                    st.markdown(titleVN.STORY, unsafe_allow_html = True)

            with col1:
                st.markdown(titleVN.CHEF_INFO, unsafe_allow_html=True)

                choice_way = st.radio("Chọn một", ["Tải ảnh lên", "Chọn từ ảnh có sẵn"])

                if choice_way == "Tải ảnh lên":
                    file_img = st.file_uploader('Tải một hình ảnh về chim')

                    if file_img:
                        img = Image.open(file_img)

                else:
                    dataset_type = st.selectbox("Loại dữ liệu", data_split_names)
                    data_folder = dtype_file_structure_mapping[dataset_type]

                    selected_species = st.selectbox("Loài chim", types_of_birds)
                    available_images = load_list_of_images(all_images, data_folder, selected_species)
                    image_name = st.selectbox("Tên hình ảnh", available_images)

                    key_path = os.path.join('bird_dataset', data_folder, image_name)
                    img = cv2.imread(key_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                recipe_button = st.button('Lấy kết quả!')

            st.markdown("<hr />", unsafe_allow_html=True)

            if recipe_button:
                with st.spinner("Chờ trong giây lát..."):
                    if file_img or key_path:
                        col3, col4 = st.columns([5, 4])
                        with col3:
                            if os.path.isdir('./runs'):
                                shutil.rmtree('./runs')

                            results = get_prediction(img, model)
                            results.save()

                            st.header("Đây là kết quả phát hiện!")

                            img_res = cv2.imread('./runs/detect/exp/image0.jpg')
                            img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
                            st.image(img_res, use_column_width = True)

                            df = results.pandas().xyxy[0]
                            del df['class']
                            st.write(df)
                        
                        with col4:
                            st.header("Mô tả")

                            des = set()
                            for name_type in df['name']:
                                if name_type not in des:
                                    if name_type == 'bird':
                                        continue

                                    elif name_type == 'eagle':
                                        with st.expander("Đại Bàng"):
                                            st.markdown(titleVN.EAGLE, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)
                                    
                                    elif name_type == 'flamingo':
                                        with st.expander("Hồng Hạc"):
                                            st.markdown(titleVN.FLAMINGO, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    elif name_type == 'golden oriole':
                                        with st.expander("Vàng Anh"):
                                            st.markdown(titleVN.GOLDEN_ORIOLE, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    elif name_type == 'hyacinth macaw':
                                        with st.expander("Vẹt Đuôi Dài"):
                                            st.markdown(titleVN.HYACINTH_MACAW, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)
                                    
                                    elif name_type == 'mute swan':
                                        with st.expander("Thiên Nga"):
                                            st.markdown(titleVN.MUTE_SWAN, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    elif name_type == 'seagull':
                                        with st.expander("Hải Âu"):
                                            st.markdown(titleVN.SEAGULL, unsafe_allow_html = True)

                                            example_images = load_list_of_images(all_images, "example", name_type)
                                            examples_of_species = np.random.choice(example_images, size = 3, replace = False)
                                            images_to_show = []

                                            for exam in examples_of_species:
                                                path = os.path.join("bird_dataset/example", name_type, exam)
                                                images_to_show.append(Image.open(path))

                                            st.image(images_to_show, width = 163)

                                    des.add(name_type)

                            if not des:
                                st.info("Không có dữ liệu để mô tả!")

                    else:
                        st.error('Không có hình ảnh nào. Vui lòng chọn một hình ảnh về chim!')

if __name__ == '__main__':
    main()