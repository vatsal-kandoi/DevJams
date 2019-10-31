document.getElementById('search-btn').addEventListener('click', () => {
    // Switch with regex
    if(document.getElementById('search-inp').value == '' || document.getElementById('search-inp').value == undefined) {
        document.getElementsByClassName('error')[0].innerHTML=`
            <span class="close">&times</span>
            Please enter a valid URL.
        `
        document.getElementsByClassName('error')[0].style.top='0px';
        setTimeout(()=> {document.getElementsByClassName('error')[0].style.top='-250px'}, 2000)
    } else {
        window.location.href=`/result.html?item=${document.getElementById('search-inp').value}`
    }
});
document.getElementsByClassName('close')[0].addEventListener('click',()=>{document.getElementsByClassName('error')[0].style.top='-250px'})