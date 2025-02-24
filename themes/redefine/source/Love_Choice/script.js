// DOM 元素获取
const yesButton = document.getElementById('yes');
const noButton = document.getElementById('no');
const questionText = document.getElementById('question');
const mainImage = document.getElementById('mainImage');

// 状态变量
let clickCount = 0;

// 拒绝按钮的文案配置
const noTexts = [
    '？你认真的吗…',
    '要不再想想？',
    '不许选这个！',
    '我会很伤心…',
    '不行:('
];

// 图片路径配置
const images = {
    shocked: 'images/shocked.png',
    think: 'images/think.png',
    angry: 'images/angry.png',
    crying: 'images/crying.png'
};

// 拒绝按钮点击事件处理
noButton.addEventListener('click', () => {
    clickCount++;

    // 放大确认按钮
    const yesSize = 1 + (clickCount * 1.2);
    yesButton.style.transform = `scale(${yesSize})`;

    // 移动拒绝按钮
    const noOffset = clickCount * 50;
    noButton.style.transform = `translateX(${noOffset}px)`;

    // 上移图片和问题文字
    const moveUp = clickCount * 25;
    mainImage.style.transform = `translateY(-${moveUp}px)`;
    questionText.style.transform = `translateY(-${moveUp}px)`;

    // 更新拒绝按钮文案
    if (clickCount <= 5) {
        noButton.innerText = noTexts[clickCount - 1];
    }

    // 更新图片
    updateImage();
});

// 确认按钮点击事件处理
yesButton.addEventListener('click', () => {
    document.body.innerHTML = `
        <div class="yes-screen">
            <h1 class="yes-text">!!!喜欢你!! ( >᎑<)♡︎ᐝ</h1>
            <img src="images/hug.png" alt="拥抱" class="yes-image">
        </div>
    `;
    document.body.style.overflow = 'hidden';
});

// 根据点击次数更新图片
function updateImage() {
    switch(clickCount) {
        case 1:
            mainImage.src = images.shocked;
            break;
        case 2:
            mainImage.src = images.think;
            break;
        case 3:
            mainImage.src = images.angry;
            break;
        case 4:
        case 5:
            mainImage.src = images.crying;
            break;
    }
}