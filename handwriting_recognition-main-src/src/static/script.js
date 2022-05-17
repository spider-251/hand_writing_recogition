// var loadFile = function (event) {
//   var image = document.getElementById('output')
//   image.src = URL.createObjectURL(event.target.files[0])
// }

var im = document.getElementById('ipimg') // or select based on classes
// im.onerror = function () {
//   // image not found or change src like this as default image:
//   im.style.display = 'none'
//   //   im.src = 'new path'
// }
console.log(im)
console.log(im.src)
if (im.src != 'abc') {
  console.log('hellol')
<<<<<<< HEAD
  im.style.display = 'block'
=======
>>>>>>> 6c7b6ad (image box updated commit)
}
