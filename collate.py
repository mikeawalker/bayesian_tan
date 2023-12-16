
files = ["title.html", "intro.html","part1.html" , "part2.html", "outro.html"]

merged_content = ""
for file in files:
    with open( file ,'rt') as f:
        content = f.read()
        merged_content += content
#Write the merged content to a new HTML file
with open('report.html', 'w') as output_file:
    output_file.write(merged_content)