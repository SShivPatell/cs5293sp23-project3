from project3 import predict, readPdf

# Check if cluster is being predicted correctly for Tucson
def test_Predict(contentsPath = 'smartcity/AZ Tucson.pdf'): 
    
    content = readPdf("AZ Tucson", contentsPath)
    cluster, df = predict(content)
    
    # Check if cluster being predicted matches that one already predicted in smartcity_eda.tsv
    assert cluster == 1