from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
	return { "message" : "Hello World" }


if __name__ == '__main__':
    import uvicorn
    
    app_str = 'main:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)