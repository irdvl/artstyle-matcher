from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

learner_file_url = 'https://drive.google.com/uc?export=download&id=1UkN-sdCPX76a1H9enAoT1nvxuEw8Grrf'
learner_file_name = 'export'
classes = [
    'andys_eyes', 
    'tyswills',
    'blxxp',
    'brandonlavender',
    'kyle_jeffers',
    'inizziv',
    'adam_brei',
    'fabvp',
    'natalie_santafe',
    'jcrecc',
    'jokalinow',
    'jasonbystrom',
    'alxbtz_',
    'domoa',
    'matekout',
    'lll.lll.lll.ll.ll.ll.l.l.l',
    'weird.sink',
    'heycaptainjane',
    'axelswanmaldini',
    'nicholas_mellefont_',
    'toscanology',
    'robin_ek',
    'ingunfosli',
    'alicehines',
    'heysupersimi',
    'mabota11',
    'toscanology',
    'jonsetter',
    'adela.llanas',
    'theharrisonford'
]
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(learner_file_url, path/f'{learner_file_name}.pkl')
    return load_learner(path)

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': 'https://instagram.com/'+str(learn.predict(img)[0])})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=5042)

