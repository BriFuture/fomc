import xml.etree.ElementTree as ET
import xml.dom.minidom as md
def prettify_xml(element):
    """格式化 XML 使其具有正确的缩进"""
    rough_string = ET.tostring(element, encoding="utf-8")
    reparsed = md.parseString(rough_string)
    formatted_xml = reparsed.toprettyxml(indent="  ")
    # return reparsed.toprettyxml(indent="  ")
    # 去除空行
    cleaned_xml = "\n".join([line for line in formatted_xml.splitlines() if line.strip()])
    return cleaned_xml

import hashlib
import base64
def hash_names(ids_names: list[str]):
    txt = ",".join(ids_names)
    hash = hashlib.sha256(txt.encode())
    dig = hash.digest()
    dat = base64.b64encode(dig).decode()
    return dat
