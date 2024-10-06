import requests
import pandas as pd
from urllib.parse import quote
import time
import hashlib


def trans_date(v_timestamp):
    """10位时间戳转换为时间字符串"""
    timeArray = time.localtime(v_timestamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def get_response(url, data):
    headers = {
        'Cookie': "buvid3=90E453C0-7883-4634-D652-2541DB8D144D84828infoc; buvid_fp_plain=undefined; LIVE_BUVID=AUTO6716518234674935; rpdid=|(u)~m~J~|Rk0J'uYY)~l~YYl; buvid4=B2829ACF-14FA-4EE1-B846-055F8D47875595980-022012815-rC57Zta57o2T5BJyYLk+AA==; header_theme_version=CLOSE; b_nut=100; enable_web_push=DISABLE; CURRENT_FNVAL=16; FEED_LIVE_VERSION=V_WATCHLATER_PIP_WINDOW2; CURRENT_BLACKGAP=0; _uuid=1613D7E7-4FEB-10448-453C-C1081DD4BD39142345infoc; DedeUserID=518532986; DedeUserID__ckMd5=7b9d3c133f7416fa; PVID=2; is-2022-channel=1; hit-dyn-v2=1; fingerprint=76c98ffc8ad43581ed5f464270259f1f; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MjY2NDE0MTEsImlhdCI6MTcyNjM4MjE1MSwicGx0IjotMX0.x0brZBOxqWCf14fayGc6IhBBZAerEHr_y7KUe0llfTU; bili_ticket_expires=1726641351; SESSDATA=bd6e9271,1741935872,64fac*92CjAsb11fN_-iixQbFTWJMEaGJpZlDIvk21nRhVptMhCsyvbO9diEXtKjMDzkiJPluGkSVnFnWlNFVXoyU1BnUVZOMzdadmcwVTdhWkFYNWVTUDJ6Mk5EYkQ1c3EzRURGTjlKZEd2cmVWdkNXaDRHamNvbnZhczVRNTZOUkNPdVdHd0M4V21BMXJBIIEC; bili_jct=b75178068db6b82c7ef3f129ed91f2b5; sid=6o8js7mi; home_feed_column=4; browser_resolution=786-918; buvid_fp=76c98ffc8ad43581ed5f464270259f1f; bp_t_offset_518532986=978059998573625344; b_lsid=10C1726AD_191FF229388",
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.0.10191 SLBChan/103 '
    }
    response = requests.get(url, params=data, headers=headers)
    return response


def get_content(oid, date, nextpage, w_rid):
    link = 'https://api.bilibili.com/x/v2/reply/wbi/main'
    params = {
        'oid': '%s' % oid,
        'type': '1',
        'mode': '3',
        'pagination_str': '{"offset":%s}' % nextpage,
        'plat': '1',
        'seek_rpid': '',
        'web_location': '1315875',
        'w_rid': w_rid,
        'wts': date
    }
    response = get_response(link, params)
    data_list = response.json()['data']['replies']
    comment_list = []
    time_list = []
    like_list = []

    # 循环爬取每一条评论数据
    for a in data_list:
        # 评论内容
        comment = a['content']['message'].replace("\n", "")
        comment_list.append(comment)
        time = trans_date(a['ctime'])
        time_list.append(time)
        like = a['like']
        like_list.append(like)

    # 把列表拼装为DataFrame数据
    df = pd.DataFrame({
        '评论时间': time_list,
        '点赞数': like_list,
        '评论内容': comment_list,
    })
    next_offset = response.json()['data']['cursor']['pagination_reply']['next_offset']

    return df, next_offset


def Hash(oid, date, nextpage):
    NextPage = '{"offset":%s}' % nextpage
    NextPage = quote(NextPage)
    en = [
        "mode=3",
        f"oid={oid}",
        f"pagination_str={NextPage}",
        "plat=1",
        "seek_rpid=",
        "type=1",
        "web_location=1315875",
        f"wts={date}"
    ]
    Jt = '&'.join(en)
    string = Jt + "ea1db124af3c7062474693fa704f4ff8"
    MD5 = hashlib.md5()
    MD5.update(string.encode('utf-8'))
    w_rid = MD5.hexdigest()
    return w_rid


class GetPopularComments():
    def __init__(self):
        self.headers = {
            "Origin": "https://www.bilibili.com",
            "Pragma": "no-cache",
            "Referer": "https://www.bilibili.com/v/popular/all/?spm_id_from=333.1007.0.0",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/109.0.0.0 Safari/537.36 SLBrowser/9.0.0.10191 SLBChan/103",
        }
        self.url = "https://api.bilibili.com/x/web-interface/popular"

    def get_data(self, page):
        params = {
            "ps": "20",
            "pn": str(page),
        }
        oid_list = []
        response = requests.get(self.url, headers=self.headers, params=params)
        data_list = response.json()['data']['list']
        for data in data_list:
            oid_list.append(data['aid'])
        return oid_list

    def main(self):
        date = int(time.time())
        all_comments = pd.DataFrame()
        for i in range(1, 10):
            oid_list = self.get_data(i)
        for idx, oid in enumerate(oid_list):
            nextpage = '""'
            for page in range(0, 1):
                print(f"正在采集第{idx}个视频,第{page}页的评论")
                w_rid = Hash(oid, date, nextpage)
                df, nextpage = get_content(oid, date, nextpage, w_rid)
                all_comments = pd.concat([all_comments, df], ignore_index=True)
                time.sleep(1)
        all_comments.to_csv("./comments/bili_comments", mode='a+', encoding='utf_8_sig')


if __name__ == "__main__":
    comments = GetPopularComments()
    comments.main()
