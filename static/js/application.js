var imageInput = document.getElementById('image')
  var similarityResult = document.getElementById('similarityResult')

  imageInput.addEventListener('input', (event) => { handleFilePush(event) })

  handleFilePush = (event) => {
    const reader = new FileReader()
    var file     = imageInput.files[0]
    var img      = document.createElement("img")
    
    img.file = file

    document.getElementById('filePreview').appendChild(img)
    reader.onload = (e) => { img.src = e.target.result }
    reader.readAsDataURL(file)
  }

  submitForm = (_event) => {
    var data = new FormData()
    data.append('image', imageInput.files[0])

    fetch('/similarity', {
      method: 'POST',
      body: data
    }).then((response) => response.json())
      .then((data) => {
        data.forEach((src) => {
          img = document.createElement("img")
          img.src =`/static/${src}`
          similarityResult.appendChild(img)
        })
      });
  }