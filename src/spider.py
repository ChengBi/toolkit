from selenium import webdriver
driver=webdriver.Firefox()
import urllib

for i in range(1600,3000):
    try:
        driver.get("http://www.paperweekly.site/papers/"+str(i))
    #     driver.find_element_by_class_name('alert').click()
        inputs = driver.find_element_by_class_name('link')
        url = inputs.text
        title = driver.find_element_by_tag_name('h1')
        filename = '_'.join(title.text.replace(':', '_').replace('\'', '').split())
    #         if filename != '':
    #     print(filename)
        if filename != '':
            print('#', i, ' => ', filename)
            urllib.request.urlretrieve(url, filename='../output/'+filename+'.pdf', reporthook=None, data=None)
    except:
        continue
#     except:
#         continue
driver.quit()
