import streamlit as st
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image, ImageDraw
from collections import defaultdict
import pandas as pd
from PIL import Image
from itertools import combinations
from streamlit_image_coordinates import streamlit_image_coordinates

def rect_dimensions(points):
    """
    points (int) -- output from cv2.boxPoints(rect) points, 4 potins are ordered in clockwise,starting from the
    upper left
    return (int) -- Calculate BBox width, height and then return the max (width, height)
    """
    width = math.sqrt((points[0,0]-points[1,0])**2 +  (points[0,1]-points[1,1])**2)
    height = math.sqrt((points[1,0]-points[2,0])**2 +  (points[1,1]-points[2,1])**2)
    return max(int(width), int(height))
    #return math.sqrt(width**2 + height**2)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_max_distance(points):
    max_distance = 0
    combins = [c for c in combinations(range(4), 2)]
    for com in combins:
        point1 = points[com[0]]
        point2 = points[com[1]]
        distance = calculate_distance(point1, point2)
        if distance >= max_distance:
            max_distance = distance
    return max_distance

def click_button_scale():
    st.session_state.button_scale = not st.session_state.button_scale

def click_button_inclusion():
    st.session_state.button_inclusion = not st.session_state.button_inclusion

def page_home():
    st.title('K模夹杂自动识别')
    st.write("V4.0")
    st.markdown(
    """
    需用户加载待检测图片并提供标尺像素比，如果像素比未知，请自行选取加载 \n
    数据表 - -  展示夹杂的统计数据    
    图片 - - 展示识别结果
    """
    )
    
    st.session_state['image_toShow'] = []
    st.session_state['image_toYolo'] = []
    st.session_state['image_name'] = []
    st.session_state['scale'] = 0

    upload_imgs = st.file_uploader("步骤一：选择待检测图片",accept_multiple_files = True )
    if len(upload_imgs) < 1 :
        st.warning("请加载图片")
        
    opencv_image_ = []
    # obatain file names
    for upload_img in upload_imgs:
        file_bytes = np.asarray(bytearray(upload_img.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.session_state['image_toShow'].append(opencv_image.copy())
        opencv_image_.append(opencv_image)
        st.session_state['image_name'].append(upload_img.name)

    #st.write(len(st.session_state['image_name']),len(st.session_state['image_toShow']))
    scale2pixel = st.text_input("步骤二：请提供标尺像素比,若未知,先归零后点击下钮",value="0")
    st.button('如果需要从图中测量？', on_click=click_button_scale)
    try:
        if scale2pixel != "0":
            scale2pixel = float(eval(scale2pixel))
            st.session_state['scale'] = scale2pixel
        elif upload_imgs and st.session_state.button_scale :
            scale_onImage = st.text_input("请提供待检测图像中的标尺值"+"("+chr(956)+"m)")
            col1, col2 = st.columns(2)
            try:
                with col1:
                    scale_onImage = int(eval(scale_onImage))
                    selected_name = st.selectbox("选择一张图片预览",st.session_state['image_name'])
                    selected_name_idx = st.session_state['image_name'].index(selected_name)
                    img_1st = opencv_image_[selected_name_idx].copy()
                    img_1st_rgb = cv2.cvtColor(img_1st,cv2.COLOR_BGR2RGB)
                    img_1st_rgb_draw = Image.fromarray(img_1st_rgb)
                    st.write("请单击下图标尺的左右两端")
                    value = streamlit_image_coordinates(img_1st_rgb_draw,use_column_width = "always")
                    if value is not None:
                        point = value["x"], value["y"]
                        if point not in st.session_state["points"]:
                            st.session_state["points"].append(point) 

                    # debug
                    # st.write(len(st.session_state["points"]))
                    # st.write(st.session_state["points"])    
                    st.write("""--------------------------------------------------------------------------------原图--------------------------------------------------------------------------------""")
                with col2:
                    st.write("自动保存并显示所选择的位置,如果需重新选择,请单击左图任意远离标尺的位置两次,清空之前标记后再重新标记")
                    for pt in st.session_state["points"][-2:]:
                        point_resize = int(pt[0]/value['width']*img_1st_rgb_draw.size[0]), int(pt[1]/value['height']*img_1st_rgb_draw.size[1])
                        circleShow =  cv2.circle(img_1st_rgb,point_resize,7,(0,0,255),-1)
                    st.session_state['scale'] = scale_onImage/abs(st.session_state['points'][-1][0]/value['width']*img_1st_rgb_draw.size[0]-st.session_state['points'][-2][0]/value['width']*img_1st_rgb_draw.size[0])
                    st.write(f"图中标尺为:{scale_onImage}",chr(956),"m", f", 对应的像素长度为:{int(abs(st.session_state['points'][-1][0]/value['width']*img_1st_rgb_draw.size[0]-st.session_state['points'][-2][0]/value['width']*img_1st_rgb_draw.size[0]))}") 
                    st.write(f"标尺的像素比为:{scale_onImage}/{int(abs(st.session_state['points'][-1][0]/value['width']*img_1st_rgb_draw.size[0]-st.session_state['points'][-2][0]/value['width']*img_1st_rgb_draw.size[0]))} = {round(st.session_state['scale'],2)}") 
                    st.image(circleShow,channels="BGR") 
                    st.write("""-------------------------------------------------------------------------------标识效果------------------------------------------------------------------------------""")
            except:
                st.warning("还未提供标尺像素比") 
        else:
            st.warning("请提供标尺像素比,若不知，请按上钮")
    except:
        st.warning("请提供标尺像素比,若不知，请按上钮")    

    for img_arry in opencv_image_:
        img = cv2.cvtColor(img_arry,cv2.COLOR_BGR2RGB)
        st.session_state['image_toYolo'] .append(Image.fromarray(img))


def page_dataframe():
    st.title('统计数据')
    #st.write(st.session_state["scale"]) 
    col1, col2 = st.columns(2)
    model_path = './weights/best.pt'
    # 提供图片路径
    model = YOLO(model_path) 
    try :
        results = model(st.session_state['image_toYolo'])
        maskCollection = []
        img_noresult_idx = []
        for res in results:
            if res.masks is not None:
                infoNames = ["name","area","length"]
                # Convert mask to single channel image
                for i in range(res.masks.data.shape[0]):
                    maskInfo = defaultdict(str)
                    mask_raw = res.masks[i].cpu().data.numpy().transpose(1, 2, 0)
                    # Convert single channel grayscale to 3 channel image
                    mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
                    # Get the size of the original image (height, width, channels)
                    h2, w2, c2 = res.orig_img.shape
                    # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
                    mask = cv2.resize(mask_3channel, (w2, h2))
                    Togray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #2D image
                    # Apply thresholding in the gray image to create a binary image
                    ret,thresh = cv2.threshold(Togray,0,255,0)
                    # Avoid data type issue
                    thresh = np.array(thresh,np.uint8)
                    # Find countour
                    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 1:
                        cnt = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)[0]
                    else:
                        cnt = contours[0]
                    left_most = tuple(cnt[cnt[:, :, 0].argmin()][0])
                    right_most = tuple(cnt[cnt[:, :, 0].argmax()][0])
                    top_most = tuple(cnt[cnt[:, :, 1].argmin()][0])
                    bottom_most = tuple(cnt[cnt[:, :, 1].argmax()][0])
                    Area = int(cv2.contourArea(cnt))
                    Area = round(Area *  st.session_state['scale'] *  st.session_state['scale'],2) # show in actual area
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int64(box)
                    Length = rect_dimensions(box)
                    #Length = calculate_max_distance([left_most,right_most,top_most,bottom_most])
                    #st.write("边界框计算的长度：",rect_dimensions(box), "极值点计算的长度：", Length)
                    Length = round(Length * st.session_state['scale'],2) # show in actual length
                    # collect all relevant info for dataframe
                    maskInfo[infoNames[0]] = st.session_state['image_name'][results.index(res)]+ f'_{i+1}'
                    maskInfo[infoNames[1]] = Area 
                    maskInfo[infoNames[2]] = Length  
                    maskCollection.append(maskInfo)
            else:
                img_noresult_idx.append(results.index(res))

        name_all = []
        area_all = []
        len_all = []

        for mC in maskCollection:
            name_all.append(mC.get("name"))
            area_all.append(mC.get("area"))
            len_all.append(mC.get("length"))

        df_data = {"Title":name_all,"Area("+chr(956)+"m" + "\u00b2" + ")":area_all,"Length("+chr(956)+"m)":len_all}
        df = pd.DataFrame(df_data)
        df["Title_"] = df["Title"].str.split("_",expand = True)[0]
        len_bins = [0,100,500,800,1200,10000]
        df["Length_level"] = pd.cut(df["Length("+chr(956)+"m)"], len_bins, labels=['xS','S', 'M', 'L','xL'])
        df.index = df.index + 1

        res = df.groupby(["Title_"],as_index=False).agg(
            {   "Area("+chr(956)+"m" + "\u00b2" + ")": ["max"],
                "Length("+chr(956)+"m)": ["max","count"]
                #"Length("+chr(956)+"m)": ["max"]
            })
        res.columns = ["Title","Max_Area("+chr(956)+"m" + "\u00b2" + ")","Max_Length("+chr(956)+"m)","Count"]

        res = pd.concat ([res,pd.DataFrame({
            "Title" : [st.session_state['image_name'][img_noresult_idx[i]] for i in range(len(img_noresult_idx))],
            "Max_Area("+chr(956)+"m" + "\u00b2" + ")" : [0] * len(img_noresult_idx),
            "Max_Length("+chr(956)+"m)" : [0] * len(img_noresult_idx),
            "Count" : [0] * len(img_noresult_idx)
        })],ignore_index = True)    
        res = res.sort_values(by='Title')
        res.reset_index(drop=True, inplace=True)
        res.index = res.index + 1

        with col1:
            st.write("统计数据")
            st.dataframe(df[["Title","Area("+chr(956)+"m" + "\u00b2" + ")","Length("+chr(956)+"m)","Length_level"]])
            st.bar_chart(df.Length_level.value_counts(), x_label="级别",y_label="数量")

        with col2:
            st.write("汇总数据")
            st.dataframe(res)
            level_count_dic = df["Length_level"].value_counts().to_dict()
            inclusion_factor = level_count_dic["xS"]*0.1+level_count_dic["S"]*0.5+level_count_dic["M"]*1+level_count_dic["L"]*1.5+level_count_dic["xL"]*2
            st.write(f"夹杂影响因子：{inclusion_factor}")
            st.write(f"夹杂率：{inclusion_factor/5}")
            st.button('注释：夹杂影响因子与夹渣率', on_click=click_button_inclusion)
            if st.session_state.button_inclusion:
                st.write("夹杂影响因子 = 各级别夹杂数 * 对应的权重")
                st.write("夹杂率 = 夹杂影响因子 /  k模断面总数(5)") 
    
    except Exception as e:
        st.write(e)
    # except:
    #     return

def page_image():
    st.title('识别结果')
    col1, col2 = st.columns(2)
     model_path = './weights/best.pt'
    # 提供图片路径
    model = YOLO(model_path) 
    try :
        results = model(st.session_state['image_toYolo'])
        imgtoshowCollection = []
        for res in results:
            if res.masks is not None:
                # Convert mask to single channel image
                cntdraw = []
                for i in range(res.masks.data.shape[0]):
                    mask_raw = res.masks[i].cpu().data.numpy().transpose(1, 2, 0)
                    # Convert single channel grayscale to 3 channel image
                    mask_3channel = cv2.merge((mask_raw,mask_raw,mask_raw))
                    # Get the size of the original image (height, width, channels)
                    h2, w2, c2 = res.orig_img.shape
                    # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
                    mask = cv2.resize(mask_3channel, (w2, h2))
                    Togray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #2D image
                    # Apply thresholding in the gray image to create a binary image
                    ret,thresh = cv2.threshold(Togray,0,255,0)
                    # Avoid data type issue
                    thresh = np.array(thresh,np.uint8)
                    # Find countour
                    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 1:
                        cnt = sorted(contours, key=lambda a: cv2.contourArea(a), reverse=True)[0]
                    else:
                        cnt = contours[0]
                    cntdraw.append(cnt)
                zeros = np.zeros((res.orig_img.shape), dtype=np.uint8)
                mask_add = cv2.fillPoly(zeros, cntdraw, color=(98, 9, 11))
                imgwithcont = 0.8 * mask_add + res.orig_img
                imgwithcont = imgwithcont.astype(np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                #st.write(len(cntdraw))
                for i in range(len(cntdraw)):
                    top_most =tuple(cntdraw[i][cntdraw[i][:, :, 1].argmin()][0])
                    cv2.putText(imgwithcont, f"Inclusion{i+1}", top_most, font, 1, (255, 0, 0), 5)
                    
                imgtoshowCollection.append(imgwithcont)
            else:
                imgtoshowCollection.append(res.orig_img)

        with col1:
            st.write("输入图片")
            for i in range(len(st.session_state['image_toShow'])):
                st.image(st.session_state['image_toShow'][i], caption=st.session_state['image_name'][i] + "_Original",channels="BGR")

        with col2:
            st.write("检测结果")           
            for i in range(len(imgtoshowCollection)):
                st.image(imgtoshowCollection[i], caption=st.session_state['image_name'][i] + "_Tested",channels="BGR")

    # except Exception as e:
    #     st.write(e)
    except:
        return

session_state = st.session_state   
if 'image_toShow' not in st.session_state:
    st.session_state['image_toShow'] = []
if 'image_toYolo' not in st.session_state:
    st.session_state['image_toYolo'] = []
if 'image_name' not in st.session_state:
    st.session_state['image_name'] = []
if 'points' not in st.session_state:
    st.session_state['points'] = []   
if 'scale' not in st.session_state:
    st.session_state['scale'] = 0
if 'button_scale' not in st.session_state:
    st.session_state.button_scale = False
if 'button_inclusion' not in st.session_state:
    st.session_state.button_inclusion = False 
if 'page' not in session_state:
    session_state['page'] = '主页'

st.set_page_config(layout="wide")
page = st.sidebar.radio('导航', ['主页','数据表','图片'])
if page == '主页':
    page_home()
elif page == '数据表':
    page_dataframe()
else:
    page_image()