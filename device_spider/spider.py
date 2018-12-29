from bs4 import BeautifulSoup
import re
import xlwt
import sys
from tqdm import tqdm
import time
import urllib.request

def url_get(url):
    req = urllib.request.Request(url)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'
    req.add_header("user-agent", user_agent)
    response_result = urllib.request.urlopen(url).read()
    html = response_result.decode('gbk')
    return html

parameters_map = {
    '主屏尺寸：':'main_screen_size',
    '主屏分辨率：':'main_screen_resolution',
    '后置摄像头：':'back_camera',
    '前置摄像头：':'front_camera',
    '电池容量：':'battery_capacity',
    '电池类型：':'battery_type',
    '核心数：':'cores_num',
    '内存：':'memory',
}

def operate(pages, output_xls_file):
    count=0
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('result', cell_overwrite_ok=True)
    sheet.write(0, 0, '编号')
    sheet.write(0, 1, '手机名称')
    sheet.write(0, 2, '参考价')
    sheet.write(0, 3, '评分')
    sheet.write(0, 4, '特点')
    sheet.write(0, 5, '详情页')
    sheet.write(0, 6, '主屏尺寸')
    sheet.write(0, 7, '主屏分辨率')
    sheet.write(0, 8, '后置摄像头')
    sheet.write(0, 9, '前置摄像头')
    sheet.write(0, 10, '电池容量')
    sheet.write(0, 11, '电池类型')
    sheet.write(0, 12, '核心数')
    sheet.write(0, 13, '内存')
    for page in range(pages):
        print('第' + str(page) + '页')
        url = 'http://detail.zol.com.cn/cell_phone_index/subcate57_0_list_1_0_1_2_0_' + str(page + 1) + '.html'
        time.sleep(2)
        text = url_get(url)
        soup = BeautifulSoup(text, 'html.parser')
        content = soup.find_all('h3')
        content = content[1: -6]
        content2 = soup.find_all('b', class_ = 'price-type')
        content3 = soup.find_all('span', class_ = 'score')
        assert len(content)==len(content2) == len(content3)
        for i in tqdm(range(len(content))):
            content_i = str(content[i])
            a = re.findall(r'(/cell_phone/index\d+.shtml)', str(content_i))
            count += 1
            #详情页
            sheet.write(count, 5, 'detail.zol.com.cn' + a[0])

            url_child = 'http://detail.zol.com.cn'+a[0]
            #time.sleep(10)
            try:
                text_child = url_get(url_child)
            except:
                text_child = ''

            soupx = BeautifulSoup(text_child, 'html.parser')
            child_info = soupx.find_all('p', class_='')

            parameters_result = {
                'main_screen_size':'',
                'main_screen_resolution':'',
                'back_camera':'',
                'front_camera':'',
                'battery_capacity':'',
                'battery_type':'',
                'cores_num':'',
                'memory':'',
            }
            for idx in range(min(len(child_info), 15)):
                try:
                    if child_info[idx].span.contents[0] in parameters_map:
                        parameters_result[parameters_map[child_info[idx].span.contents[0]]] = child_info[idx].contents[1].replace('\t', '').replace('\r', '').replace('\n', '').replace(' ', '')
                except:
                    continue
            #parameters_map
            sheet.write(count, 6, parameters_result['main_screen_size'])
            sheet.write(count, 7, parameters_result['main_screen_resolution'])
            sheet.write(count, 8, parameters_result['back_camera'])
            sheet.write(count, 9, parameters_result['front_camera'])
            sheet.write(count, 10, parameters_result['battery_capacity'])
            sheet.write(count, 11, parameters_result['battery_type'])
            sheet.write(count, 12, parameters_result['cores_num'])
            sheet.write(count, 13, parameters_result['memory'])

            #编号
            sheet.write(count, 0, str(count))
            soup2 = BeautifulSoup(content_i, 'html.parser')
            cha = soup2.h3.a.span.get_text()
            #特点
            sheet.write(count, 4, cha)
            n = re.findall(r'">.*?<', content_i)
            #手机名称
            sheet.write(count, 1, n[0][2:-1].replace('\t', '').replace('\r', '').replace('\n', '').replace(' ', ''))

            content2_i = str(content2[i])
            soup3 = BeautifulSoup(content2_i, 'html.parser')
            p = soup3.b.get_text()
            #参考价
            sheet.write(count, 2, p)

            content3_i = str(content3[i])
            soup4 = BeautifulSoup(content3_i, 'html.parser')
            s = soup4.span.get_text()
            #评分
            sheet.write(count, 3, s)

        book.save(output_xls_file)


if __name__ == '__main__':
    operate(2, r'demo.xls')
