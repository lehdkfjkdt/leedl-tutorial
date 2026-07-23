import urllib.request
import os

dest = r"d:\Dtpmh_Accumulate\dtpmh_accumulate\Professional and Academic Accumulation\learning\paper-review\ABot-N0\ABot-N0论文导读_figures"
base = "https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/static/images"

files = [
    ("fig1_overview.jpg", "overview.jpg"),
    ("fig2_architecture.jpg", "robonavi_architecture.jpg"),
    ("fig3_datasource.jpg", "datasource.jpg"),
    ("fig4_reasoning.jpg", "reasoning.jpg"),
    ("fig5_agentic.jpg", "agentic.jpg"),
    ("fig6_agent_sys.jpg", "agent_sys.jpg"),
    ("fig7_hardware.jpg", "hardware.jpg"),
]

for out_name, src_name in files:
    url = f"{base}/{src_name}"
    out_path = os.path.join(dest, out_name)
    try:
        urllib.request.urlretrieve(url, out_path)
        size = os.path.getsize(out_path)
        print(f"OK: {out_name} ({size} bytes)")
    except Exception as e:
        print(f"FAIL: {out_name} - {e}")

print("DONE")
