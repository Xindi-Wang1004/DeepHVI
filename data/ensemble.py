import pandas as pd
import requests

# 读取 CSV 文件
input_file = '1.csv'  # 请替换为你的实际文件名
df = pd.read_csv(input_file, encoding='ISO-8859-1')

# 获取第一列的所有 RefSeq ID
refseq_ids = df.iloc[:, 0].dropna().astype(str).tolist()

# Ensembl BioMart API URL
url = "https://www.ensembl.org/biomart/martservice"

# 构造查询 XML
query_template = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Filter name="refseq_mrna" value="{refseq_id}"/>
        <Attribute name="refseq_mrna"/>
        <Attribute name="ensembl_gene_id"/>
    </Dataset>
</Query>"""

# 存储转换结果
mapping = {}

# 逐个查询 RefSeq ID 并获取对应的 Ensembl Gene ID
for refseq_id in refseq_ids:
    query = query_template.replace("{refseq_id}", refseq_id)
    response = requests.get(url, params={"query": query})
    
    if response.status_code == 200:
        lines = response.text.strip().split("\n")
        if len(lines) > 1:
            _, ensembl_id = lines[1].split("\t")
            mapping[refseq_id] = ensembl_id
        else:
            mapping[refseq_id] = None
    else:
        mapping[refseq_id] = None
        print(f"Failed to fetch data for {refseq_id}")

# 将结果添加到 DataFrame 中
df["Ensembl_Gene_ID"] = df.iloc[:, 0].map(mapping)

# 保存新的 CSV 文件
output_file = "output_with_ensembl_ids.csv"
df.to_csv(output_file, index=False)

print(f"Ensembl Gene IDs saved to {output_file}")
