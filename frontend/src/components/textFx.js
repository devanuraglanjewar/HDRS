import gsap from 'gsap'

const SCRAMBLE_CHARS = '!<>-_\\/[]{}—=+*^?#________'

export function scrambleText(element, finalText, { duration = 0.9, delay = 0 } = {}) {
  if (!element) return () => {}

  const originalText = finalText ?? element.textContent ?? ''
  const state = {
    value: 0
  }

  const update = () => {
    const progress = state.value
    const visibleCount = Math.floor(originalText.length * progress)

    let output = ''
    for (let i = 0; i < originalText.length; i += 1) {
      const char = originalText[i]
      if (char === ' ') {
        output += ' '
        continue
      }

      if (i < visibleCount) {
        output += char
        continue
      }

      const charIndex = Math.floor(Math.random() * SCRAMBLE_CHARS.length)
      output += SCRAMBLE_CHARS[charIndex]
    }

    element.textContent = output
  }

  const tween = gsap.to(state, {
    value: 1,
    duration,
    delay,
    ease: 'power2.out',
    onUpdate: update,
    onComplete: () => {
      element.textContent = originalText
    }
  })

  return () => {
    tween.kill()
    element.textContent = originalText
  }
}

