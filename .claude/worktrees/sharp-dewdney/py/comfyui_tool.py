
import json
import os
import random
import time
import uuid

import urllib.parse
import urllib.request
import aiohttp
import asyncio

from py.get_setting import UPLOAD_FILES_DIR, load_settings,get_host,get_port


    
client_id = str(uuid.uuid4())

def queue_prompt(prompt,server_address,settings):
    api_key = ""
    if settings:
        if settings["comfyuiAPIkey"]:
            api_key = settings["comfyuiAPIkey"]
    if api_key:
        p = {
            "prompt": prompt, 
            "client_id": client_id ,
            "extra_data": 
            {
                "api_key_comfy_org": api_key,
            },
        }
    else:
        p = {
            "prompt": prompt, 
            "client_id": client_id,
        }
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request("{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type,server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("{}/view?{}".format(server_address, url_values)) as response:
        return response.read()


def get_history(prompt_id,server_address):
    with urllib.request.urlopen("{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())
    
def get_all(prompt,server_address,settings):
    HOST = get_host()
    if HOST == '0.0.0.0':
        HOST = '127.0.0.1'
    PORT = get_port()
    prompt_id = queue_prompt(prompt,server_address,settings)["prompt_id"]
    image_path_list = []
    while True:
        try:
            history = get_history(prompt_id,server_address)[prompt_id]
            break
        except Exception:
            time.sleep(1)
            continue

    for o in history["outputs"]:
        for node_id in history["outputs"]:
            node_output = history["outputs"][node_id]
            if "images" in node_output:
                for image in node_output["images"]:
                    image_data = get_image(image["filename"], image["subfolder"], image["type"],server_address)
                    with open(os.path.join(UPLOAD_FILES_DIR, image["filename"]), "wb") as f:
                        f.write(image_data)
                    image_url = f"http://{HOST}:{PORT}/uploaded_files/{image['filename']}"
                    image_path_list.append(image_url)
    return image_path_list




async def upload_image_via_url(image_url, server_address):
    upload_url = f"{server_address}/upload/image"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            # 下载图片
            async with session.get(image_url) as response:
                if response.status != 200:
                    print("Failed to download image.")
                    return None
                image_content = await response.read()

            # 构造 multipart/form-data 请求体
            filename = image_url.split('/')[-1]
            mpwriter = aiohttp.MultipartWriter('form-data')
            
            # 添加普通字段
            mpwriter.append('false', {'content-type': 'text/plain'})  # overwrite
            mpwriter.append('input', {'content-type': 'text/plain'})  # type
            mpwriter.append('', {'content-type': 'text/plain'})       # subfolder

            # 添加文件字段
            part = mpwriter.append(image_content)
            part.set_content_disposition('form-data', name='image', filename=filename)

            # 上传图片
            headers = {
                'Content-Type': mpwriter.content_type,
            }
            async with session.post(upload_url, data=mpwriter, headers=headers) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    print("Image uploaded successfully:", result)
                    return result
                else:
                    print(f"Failed to upload image: {resp.status}")
                    print(await resp.text())
                    return None
    except Exception as e:
        print("Upload error:", str(e))
        return None

running_comfyuiServers = []

async def comfyui_tool_call(tool_name, text_input = None, image_input = None,text_input_2 = None,image_input_2 = None):
    settings = await load_settings()
    comfyuiServers = settings["comfyuiServers"]
    server_address = ""
    # 如果没有找到可用的服务器，则则等待直到找到一个可用的服务器
    count = 0
    while server_address == "" or count > 30 or comfyuiServers == []:
        await asyncio.sleep(1)
        for server in comfyuiServers:
            if server not in running_comfyuiServers:
                running_comfyuiServers.append(server)
                server_address = server
                break
        count += 1

    if server_address == "":
        return "没有可用的comfyui服务器"
    
    WF_PATH = UPLOAD_FILES_DIR + f"/{tool_name}" + ".json"
    with open(WF_PATH, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    prompt = json.loads(prompt_text)

    using_workflow = {}

    for workflow in settings["workflows"]:
        if workflow["unique_filename"] == tool_name:
            using_workflow = workflow
            break

    if using_workflow == {}:
        return "没有找到对应的workflow"

    if text_input is not None:
        text_nodeId = using_workflow["text_input"]["nodeId"]
        text_inputField = using_workflow["text_input"]["inputField"]
        prompt[text_nodeId]["inputs"][text_inputField] = text_input


    if text_input_2 is not None:
        text_nodeId = using_workflow["text_input_2"]["nodeId"]
        text_inputField = using_workflow["text_input_2"]["inputField"]
        prompt[text_nodeId]["inputs"][text_inputField] = text_input_2

    if image_input is not None:
        image_nodeId = using_workflow["image_input"]["nodeId"]
        image_inputField = using_workflow["image_input"]["inputField"]
        image_result = await upload_image_via_url(image_input, server_address)  # 使用 await
        if image_result is not None:
            image_name = image_result.get("name")
            if image_name:
                prompt[image_nodeId]["inputs"][image_inputField] = image_name
                print("Image uploaded successfully.")
            else:
                return "图片上传失败"
        else:
            return "图片上传失败"

    if image_input_2 is not None:
        image_nodeId = using_workflow["image_input_2"]["nodeId"]
        image_inputField = using_workflow["image_input_2"]["inputField"]
        image_result = await upload_image_via_url(image_input_2, server_address)  # 使用 await
        if image_result is not None:
            image_name_2 = image_result.get("name")
            if image_name_2:
                prompt[image_nodeId]["inputs"][image_inputField] = image_name_2
                print("Image uploaded successfully.")
            else:
                return "图片上传失败"
        else:
            return "图片上传失败"

    if "seed_input" in using_workflow and using_workflow["seed_input"] is not None:
        seed_nodeId = using_workflow["seed_input"].get("nodeId")
        seed_inputField = using_workflow["seed_input"].get("inputField")
        if seed_nodeId and seed_inputField:
            prompt[seed_nodeId]["inputs"][seed_inputField] = random.randint(0, 2**32 - 1)

    if "seed_input2" in using_workflow and using_workflow["seed_input2"] is not None:
        seed_nodeId = using_workflow["seed_input2"].get("nodeId")
        seed_inputField = using_workflow["seed_input2"].get("inputField")
        if seed_nodeId and seed_inputField:
            prompt[seed_nodeId]["inputs"][seed_inputField] = random.randint(0, 2**32 - 1)

    image_path_list = get_all(prompt,server_address,settings)

    image_path_list = get_all(prompt,server_address,settings)

    running_comfyuiServers.remove(server_address)

    return json.dumps({"image_path_list": image_path_list})