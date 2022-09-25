import json
import boto3
from urllib.parse import unquote_plus
import fitz


def pdfToText(event, context):

    s3 = boto3.client("s3")
    if event:
        file_obj = event["Records"][0]
        bucketname = str(file_obj["s3"]["bucket"]["name"])
        filename = unquote_plus(str(file_obj["s3"]["object"]["key"]))
        fileObj = s3.get_object(Bucket=bucketname, Key=filename)
        file_content = fileObj["Body"].read()

        with fitz.open(stream=file_content, filetype="pdf") as doc:
            text = ""
            print(doc)
            for page in doc:
                print(page)
                text += page.get_text()

        print(text)
    return {"statusCode": 200, "body": json.dumps("Text",text)}