import os
import torch
import cv2
import json
import shutil
import tempfile
import requests
from src import titleEN
from src import titleVN
import numpy as np
import streamlit as st
from PIL import Image

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache()
def load_model(path: str = 'https://www.dropbox.com/s/lx3tw9w0g9y44xj/best.pt?dl=0'):
    url = path
    r = requests.get(path, allow_redirects=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = r)
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

    #model = load_model()
    all_images = load_file_structure()
    types_of_birds = sorted(list(all_images['train'].keys()))
   
    dtype_file_structure_mapping = {
        'All Images': 'all_image',
        'Images Used To Train The Model': 'train',
        'Images Used To Tune The Model': 'valid',
        'Images The Model Has Never Seen': 'test'
    }
    data_split_names = list(dtype_file_structure_mapping.keys())

    # local_css("style.css")

    with st.sidebar:
        st.image(Image.open('imgs/Logo02.png'), width = 100)

        select_page = st.radio("SELECT PAGE", ["HOME PAGE", "ABOUT ADB", "CONTACT"])
        st.markdown("<br /><br /><br /><br /><br /><br />", unsafe_allow_html = True)
        st.markdown("<hr />", unsafe_allow_html = True)
        language = st.selectbox("LANGUAGE", ["English", "Vietnamese"])

    if language == "English":
        if select_page == "CONTACT":
            col1, col2, col3 = st.columns([3.5, 0.5, 2])

            with col1:
                st.header("CONTACT ME")
                full_name = st.text_input("Full name *")
                phone_number = st.text_input("Phone number *")
                email = st.text_input("Email *")
                note = st.text_area("Note")
            
            with col3:
                st.markdown("<br /><br /><br />", unsafe_allow_html = True)

                st.subheader("MR. SON NGUYEN")

                st.markdown("<br />", unsafe_allow_html = True)

                st.markdown("""<p>📌 Ho Chi Minh City, Vietnam</p>""".strip(), unsafe_allow_html = True)
                st.markdown("""<p>✉️ nbthanhson1604@gmail.com</p>""".strip(), unsafe_allow_html = True)
                st.markdown("""<p>📞 (+84)96.700.7661</p>""".strip(), unsafe_allow_html = True)

                st.markdown("<br />", unsafe_allow_html = True)

                st.markdown(titleEN.GITHUB_ICON, unsafe_allow_html = True)
                st.markdown(titleEN.LINKEDIN_ICON, unsafe_allow_html = True)

            col1, col2= st.columns([10, 1])
            with col1:
                send = st.button("SEND")
            with col2:
                st.image(Image.open('imgs/Logo01.png'), use_column_width = True)

            st.markdown("<hr />", unsafe_allow_html = True)

            if send:
                if not full_name:
                    st.warning("Please enter your full name!")
                elif not phone_number:
                    st.warning("Please enter your phone number!")
                elif not email:
                    st.warning("Please enter your email address!")
                else:
                    st.success("Messenger sent! Thank you ♡️")

        elif select_page == "ABOUT ADB":
            st.header("WHO I AM")

            col1, col2 = st.columns([1, 5])

            with col1:
                st.image(Image.open('imgs/ava.jpeg'), use_column_width = True)

            _, col, _ = st.columns([3, 2, 3])

            with col:
                learn_more = st.button("LEARN MORE")
            
            if learn_more:
                st.write("abd")

            st.markdown("<hr />", unsafe_allow_html = True)
            st.markdown("<br />", unsafe_allow_html = True)

            st.header("ABOUT ADB")

            col1, col2 = st.columns([5, 1.5])

            with col2:
                st.image(Image.open('imgs/Logo02.png'), use_column_width = True)

            _, coll, _ = st.columns([3, 2, 3])

            with coll:
                learn_more2 = st.button("LEARN MORE", key = 2)
            
            if learn_more2:
                st.write("abd")

        elif select_page == "HOME PAGE":
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
                                    "Upload a video", 
                                    "Choose from available images"])

                if choice_way == "Upload an image":
                    file_img = st.file_uploader("Upload an Image of Birds")

                    if file_img:
                        img = Image.open(file_img)

                elif choice_way == "Upload a video":
                    file_vid = st.file_uploader("Upload a Video of Birds")

                    if file_vid:
                        tfile = tempfile.NamedTemporaryFile(delete=False)
                        tfile.write(file_vid.read())
                        # vf = cv2.VideoCapture(tfile.name)
                        st.video(tfile.name)
                        
                        # stframe = st.empty()

                        # while vf.isOpened():
                        #     ret, frame = vf.read()
                        #     # if frame is read correctly ret is True
                        #     if not ret:
                        #         print("Can't receive frame (stream end?). Exiting ...")
                        #         break
                        #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #     stframe.image(gray)

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
        if select_page == "CONTACT":
            col1, col2, col3 = st.columns([3.5, 0.5, 2])

            with col1:
                st.header("LIÊN HỆ")
                full_name = st.text_input("Họ và tên *")
                phone_number = st.text_input("Số điện thoại *")
                email = st.text_input("Địa chỉ email *")
                note = st.text_area("Ghi chú")
            
            with col3:
                st.markdown("<br /><br /><br />", unsafe_allow_html = True)

                st.subheader("NGUYỄN BÙI THANH SƠN")

                st.markdown("<br />", unsafe_allow_html = True)

                st.markdown("""<p>📌 Thành phố Hồ Chí Minh, Việt Nam</p>""".strip(), unsafe_allow_html = True)
                st.markdown("""<p>✉️ nbthanhson1604@gmail.com</p>""".strip(), unsafe_allow_html = True)
                st.markdown("""<p>📞 (+84)96.700.7661</p>""".strip(), unsafe_allow_html = True)

                st.markdown("<br />", unsafe_allow_html = True)

                st.markdown(titleVN.GITHUB_ICON, unsafe_allow_html = True)
                st.markdown(titleVN.LINKEDIN_ICON, unsafe_allow_html = True)

            col1, col2 = st.columns([10, 1])
            with col1:
                send = st.button("GỬI")
            with col2:
                st.image(Image.open('imgs/Logo01.png'), use_column_width = True)

            st.markdown("<hr />", unsafe_allow_html = True)

            if send:
                if not full_name:
                    st.warning("Hãy điền họ và tên của bạn!")
                elif not phone_number:
                    st.warning("Hãy điền số điện thoại của bạn!")
                elif not email:
                    st.warning("Hãy điền địa chỉ email của bạn!")
                else:
                    st.success("Thông tin đã được gửi! Cảm ơn sự quan tâm của bạn ♡️")

        elif select_page == "ABOUT ADB":
            st.header("TÔI LÀ AI")

            col1, col2 = st.columns([1, 5])

            with col1:
                st.image(Image.open('imgs/ava.jpeg'), use_column_width = True)

            _, col, _ = st.columns([3, 2, 3])

            with col:
                learn_more = st.button("TÌM HIỂU THÊM")
            
            if learn_more:
                st.write("abd")

            st.markdown("<hr />", unsafe_allow_html = True)
            st.markdown("<br />", unsafe_allow_html = True)

            st.header("ADB LÀ GÌ")

            col1, col2 = st.columns([5, 1.5])

            with col2:
                st.image(Image.open('imgs/Logo02.png'), use_column_width = True)

            _, coll, _ = st.columns([3, 2, 3])

            with coll:
                learn_more2 = st.button("TÌM HIỂU THÊM", key = 2)
            
            if learn_more2:
                st.write("abd")

        elif select_page == "HOME PAGE":
            col1, col2 = st.columns([8.1, 4])
            file_img, file_vid, key_path = '', '', ''

            with col2:
                logo = Image.open('imgs/Logo01.png')
                st.image(logo, use_column_width = True)

                with st.expander("Cách sử dụng ADB?", expanded = True):
                    st.markdown(titleVN.STORY, unsafe_allow_html = True)

            with col1:
                st.markdown(titleVN.CHEF_INFO, unsafe_allow_html=True)

                choice_way = st.radio("Chọn một", ["Tải ảnh lên", "Tải video lên", "Chọn từ ảnh có sẵn"])

                if choice_way == "Tải ảnh lên":
                    file = st.file_uploader('Tải một hình ảnh về chim')

                    if file:
                        img = Image.open(file)

                elif choice_way == "Tải video lên":
                    file_vid = st.file_uploader("Tải một video về chim")

                    if file_vid:
                        tfile = tempfile.NamedTemporaryFile(delete = False)
                        tfile.write(file_vid.read())
                        # vf = cv2.VideoCapture(tfile.name)
                        st.video(tfile.name)

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
